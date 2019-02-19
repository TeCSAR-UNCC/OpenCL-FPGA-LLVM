#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>

#define SHORTLIM    2
#define MEDLIM      6

typedef enum stride_pattern {SHORT, MEDIUM, LONG, VARIABLE, RANDOM} Stride;

struct Variable {
    char* instruction[256];
    char* varType;
    char* varName;
    char* dependency[256];
    char* dependency2[256];
    bool predictable;
    Stride stride;
    
    int instIndex;
    int depNum;
    bool localScope;
};

struct Function {
    char* funcName;
    char* params;
    struct Variable vars[64];
    int varIndex;
    int localCount;
    int globalCount;
} functions[32];

bool verbose = false;
bool debug = false;

bool
isPredictable(FILE* file, char* funcName, char* instNumber, int funcIndex, int varIndex, int instIndex);

bool
alreadyPredicted(char* function, char* variable);

Stride
findStride(FILE* file, int fxnnum, int varnum);

Stride
numericStrideFinder(struct Variable var, int depnum);

Stride
loopIndexStrideFinder(FILE* file, char* fxnname, char* index);

void
printGlobals(FILE* outfile, int funcIndex, int varIndex);

void
printLocals(FILE* outfile, int funcIndex, int varIndex);

int
main(int argc, char* argv[]) {
    
    FILE* infile;
    FILE* infileStart;
    FILE* outfile;
    char* inputFileName;
////////////////////////////////////////////////////////////////////////////////////////////
/*
 * File Operations
 */
////////////////////////////////////////////////////////////////////////////////////////////    
    if(argc == 2) {
        if(!strcmp(argv[1], "--flags")) {
            printf("Recognized flags:\n\t-v or --verbose\n");
            return 0;
        }
        inputFileName = argv[1];
    } else if (argc > 2) {
        inputFileName = argv[2];
        char* flags = argv[1];
        if(!strcmp(flags, "-v") || !strcmp(flags, "--verbose")) {
            verbose = true;
        } else if(!strcmp(flags, "-d") || !strcmp(flags, "--debug")) {
            debug = true;
        } else if(!strcmp(flags, "-dv") || !strcmp(flags, "-vd")) {
            verbose = true;
            debug = true;
        } else {
            printf("Unrecognized flag. Try running ./parse --flags\n");
        }
    } else {
        printf("I require an input file as an argument!\n");
        return 0;
    }
    
    if(!strstr(inputFileName, ".ll"))
    {
        printf("Input File of Wrong Format!\n");
        return 0;
    }
    if(verbose) printf("Reading: %s\n", inputFileName);
    infile = fopen(inputFileName, "r");
    infileStart = fopen(inputFileName, "r");

    if(infile == NULL) {
        printf("Could not find specified file!\n");
        return 0;
    }
    
    char *slash = inputFileName, *next;
        
    while ((next = strpbrk(slash + 1, "/"))) slash = next;
    if (inputFileName != slash) slash++;
    char* path = strndup(inputFileName, slash - inputFileName);
    char* infilename = strdup(slash);

    char* fileparse = strstr(infilename, ".ll");
    char* outfilename = strcat(strndup(infilename, fileparse - infilename), ".parsed");
    
    char* outputFileName = malloc(strlen(path) + strlen(outfilename) + 1);
    strcpy(outputFileName, path);
    strcat(outputFileName, outfilename);
    outfile = fopen(outputFileName, "w+");
    fputs(infilename, outfile);
    fputs("\n", outfile);
////////////////////////////////////////////////////////////////////////////////////////////
    
    int funcIndex = -1;
    int varIndex;
    
    char line[256];
    
    while (fgets(line, sizeof(line), infile)) {
    
////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Search for a function and increment the function index if one is found.
 */
////////////////////////////////////////////////////////////////////////////////////////////
        char* functionSearch = strstr(line, "efine ");
        
        if(functionSearch)
        {
            char* argSearch = strstr(line, "(");
            char* paramSearch = strstr(argSearch, ")") + 1;
            functionSearch += 6;
            funcIndex++;
            functions[funcIndex].funcName = strndup(functionSearch, argSearch - functionSearch);
            functions[funcIndex].params = strndup(argSearch, paramSearch - argSearch);
            if(verbose) printf("Identified a function: %s Searching for variables\n", functions[funcIndex].funcName);
            varIndex = -1;
            functions[funcIndex].localCount = 0;
            functions[funcIndex].globalCount = 0;
        }
        
////////////////////////////////////////////////////////////////////////////////////////////        
/*
 * Search for Pointers and parse their parameters
 */
////////////////////////////////////////////////////////////////////////////////////////////
        char* lineSearch = strstr(line, "getelementptr");
        
        if(lineSearch && (funcIndex >= 0))
        {
            char* instSearch = strstr(line, "%");
            char* instSearchEnd = strstr(instSearch, " ");
            char* instruction = strndup(instSearch, instSearchEnd - instSearch + 1);
            lineSearch += 23;
            char* typeSearch = strstr(lineSearch, "* ");
            typeSearch++;
            char* type = strndup(lineSearch, typeSearch - lineSearch);
            typeSearch++;
            
            char* commaSearch = strstr(typeSearch, ",");
            char* pointer = strndup(typeSearch, commaSearch - typeSearch);
            lineSearch = commaSearch + 2;
            char* index;
            char* index2;
            int depNum;
            char* commaSearch2 = strstr(lineSearch, ",");
            if(commaSearch2) {
                index = strndup(lineSearch, commaSearch2 - lineSearch);
                lineSearch = commaSearch2 + 2;
                
                char* commaSearch3 = strstr(lineSearch, ",");
                if(commaSearch3) {
                    index2 = strndup(lineSearch, commaSearch3 - lineSearch);
                    depNum = 2;
                } else {
                    index2 = "\0";
                    depNum = 1;
                }
            } else {
                char* nullSearch = strstr(lineSearch, "\n\0");
                index = strndup(lineSearch, nullSearch - lineSearch);
                index2 = "\0";
                depNum = 1;
            }
            if(verbose) printf("Identified a variable: %s \n", pointer);
////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Check to see if the pointer is part of the parameters of the function
 * If yes, check to see if the pointer has already been indentified
 * Add new info to pointer if it already exists
 * Otherwise create a new pointer variable
 */
////////////////////////////////////////////////////////////////////////////////////////////            
            bool localOnly;
            
            if(strstr(functions[funcIndex].params, pointer)) {
                localOnly = false;
            } else {
                localOnly = true;
            }
            int i = 0;
            if(functions[funcIndex].vars[0].varName) {
                while((strcmp(pointer, functions[funcIndex].vars[i].varName) != 0) && i <= varIndex) {
                    if(i < varIndex) i++;
                    else break;
                }
            } else {
                i = 64;
            }
            if((functions[funcIndex].vars[0].varName) && (strcmp(pointer, functions[funcIndex].vars[i].varName) != 0)) {
                i = 64;
            }
            
            if(i <= varIndex) {
                int instIndex = functions[funcIndex].vars[i].instIndex;
                functions[funcIndex].vars[i].instruction[instIndex] = instruction;
                functions[funcIndex].vars[i].varType = type;
                functions[funcIndex].vars[i].varName = pointer;
                functions[funcIndex].vars[i].dependency[instIndex] = index;
                functions[funcIndex].vars[i].dependency2[instIndex] = index2;
                if(verbose) printf("Identified memory access of pointer: %s\nIdentifying access type\n", 
                    functions[funcIndex].vars[i].varName);
                functions[funcIndex].vars[i].predictable = functions[funcIndex].vars[i].predictable && isPredictable(infileStart, 
                            functions[funcIndex].funcName, 
                            functions[funcIndex].vars[i].instruction[instIndex], 
			    funcIndex, i, instIndex);
                functions[funcIndex].vars[i].localScope = localOnly;
                functions[funcIndex].vars[i].instIndex++;
            } else {
                if(verbose) printf("New variable found\n");
                varIndex++;
                functions[funcIndex].varIndex = varIndex;
                functions[funcIndex].vars[varIndex].instIndex = 0;
                int instIndex = functions[funcIndex].vars[varIndex].instIndex;
                functions[funcIndex].vars[varIndex].instruction[instIndex] = instruction;
                functions[funcIndex].vars[varIndex].varType = type;
                functions[funcIndex].vars[varIndex].varName = pointer;
                functions[funcIndex].vars[varIndex].dependency[instIndex] = index;
                functions[funcIndex].vars[varIndex].dependency2[instIndex] = index2;
                functions[funcIndex].vars[varIndex].depNum = depNum;
                if(verbose) printf("Identified memory access of pointer: %s at instruction: %s\nIdentifying access type\n", 
                    functions[funcIndex].vars[varIndex].varName, functions[funcIndex].vars[varIndex].instruction[instIndex]);
                functions[funcIndex].vars[varIndex].predictable = true; //temporary value for first check
                functions[funcIndex].vars[varIndex].predictable = isPredictable(infileStart, 
                            functions[funcIndex].funcName, 
                            functions[funcIndex].vars[varIndex].instruction[instIndex], 
			    funcIndex, varIndex, instIndex);
                functions[funcIndex].vars[varIndex].localScope = localOnly;
                if(localOnly) {
                    functions[funcIndex].localCount++;
                } else {
                    functions[funcIndex].globalCount++;
                }
                functions[funcIndex].vars[varIndex].instIndex++;
            }
////////////////////////////////////////////////////////////////////////////////////////////
        }
        rewind(infileStart);
    }
    int i, j, k;
    
////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Check the stride pattern of the pointer
 */
////////////////////////////////////////////////////////////////////////////////////////////
    if(verbose) printf("Checking Stride Patterns\n");
    for(i = 0; i <= funcIndex; i++) {
        if(debug) printf("Fxn: %d\n", i);
        for(j = 0; j <= functions[i].varIndex; j++) {
            if(debug) printf("\tVar: %d\n", j);
            if(functions[i].vars[j].predictable) {
                functions[i].vars[j].stride = findStride(infileStart, i, j);
            } else {
                functions[i].vars[j].stride = RANDOM;
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Print Function and Pointer data to File
 */
////////////////////////////////////////////////////////////////////////////////////////////    
    for(i = 0; i <= funcIndex; i++) {
        fprintf(outfile, "\nFunction: %s\n", functions[i].funcName);
        fprintf(outfile, "Parameters: %s\n", functions[i].params);
        
        if(functions[i].vars[0].varName) {
            if(functions[i].globalCount > 0) {
                fprintf(outfile, "\n\tGlobal Pointers:\n");
                for(j = 0; j <= functions[i].varIndex; j++) {
                    if(!functions[i].vars[j].localScope) {
                        printGlobals(outfile, i, j);
                    }
                }
            }
            if(functions[i].localCount > 0) {
                fprintf(outfile, "\n\tLocal Pointers:\n");
                for(j = 0; j <= functions[i].varIndex; j++) {
                    if(functions[i].vars[j].localScope) {
                        printLocals(outfile, i, j);
                    }
                }
            }
        } else {
            fprintf(outfile, "\n\tNo indexed pointers\n");
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////
    
    fclose(infile);
    fclose(infileStart);
    fclose(outfile);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Check if a variable is statically predictable
 * Recursively searches for any undefined variables in an instruction
 * Ignores previously predicted variables
 *
 * @return lastbool The overall predictability of a variable from static behaviour
 */
////////////////////////////////////////////////////////////////////////////////////////////
bool
isPredictable(FILE* file, char* funcName, char* instNumber, int funcIndex, int varIndex, int instIndex) {
    bool inFunction = false;
    bool lastbool = true;
    char line[256];
    if(verbose) printf("\tChecking for instruction: %s\n\t in function: %s\n", instNumber, funcName);
    
    while (fgets(line, sizeof(line), file)) {
        if (!inFunction) {
            char* functionSearch = strstr(line, "efine ");
            if(functionSearch) {
                if(verbose) printf("\tFound a function! ");
                char* argSearch = strstr(line, "(");
                functionSearch += 6;
                if(verbose) printf("%s\n", strndup(functionSearch, argSearch - functionSearch));
                if(strcmp(funcName, strndup(functionSearch, argSearch - functionSearch)) == 0) {
                    inFunction = true;
                    if(verbose) printf("\tFound %s\n", funcName);
                }
            }
        } else {
            char* findinst = strstr(line, instNumber);
            if (findinst && (findinst <= line + 3)) {
                if((strcmp(instNumber, strndup(findinst, strlen(instNumber))) == 0)) {
                    if(verbose) printf("\tFound %s\t%s\n", instNumber, line);
                    findinst++;
                    
                    char* varSearch = strstr(findinst, "%");
                    if(!varSearch) {
                        if(verbose) printf("No variables found\n");
			if(strstr(findinst, "@get_global_id")) {
			    functions[funcIndex].vars[varIndex].dependency[instIndex]="tid";
			    return true;
			}
                    } else {
                        char* ptrSearch = strstr(line, "*");
                        if(ptrSearch && (ptrSearch > varSearch)) {
                            varSearch = strstr(ptrSearch, "%");
                        }
                        bool moreVars;
                        do {                            
                            char* commaSearch = strstr(varSearch, ",");
                            char* foundvar;
                            if(commaSearch) {
                                foundvar = strndup(varSearch, strstr(varSearch, ",") - varSearch);
                            } else {
                                char* spaceSearch = strstr(varSearch, " ");
                                    if(spaceSearch) {
                                        foundvar = strndup(varSearch, strstr(varSearch, " ") - varSearch);
                                    } else {
                                        foundvar = strndup(varSearch, strstr(varSearch, "\n") - varSearch);
                                    }
                            }
                            varSearch++;
                            if(isdigit(*varSearch)) {
                                if(verbose) printf("**Found another instruction**\n");
                                rewind(file);
                                lastbool = lastbool && isPredictable(file, funcName, foundvar, funcIndex, varIndex, instIndex);
                            } else if(!strcmp(strndup(varSearch, 3), "ind")) { //Checking if the variable is a loop index
                                //if(debug) printf("Found loop index\n");
                                lastbool = lastbool && true; //Sanity check
                            } else if(alreadyPredicted(funcName, foundvar)) {
                                //if(debug) printf("Found previously resolved variable\n");
                                lastbool = lastbool && true; //Sanity check
                            } else {
                                //if(debug) printf("Found unresolved variable: %s\n", foundvar);
                                lastbool = false;
                            }
                            varSearch += 3;
                            char* newvarSearch = strstr(varSearch, "%");
                            if(newvarSearch) {
                                moreVars = true;
                                varSearch = newvarSearch;
                            }
                            else moreVars = false;
                        } while(moreVars);
                    }      
                }
                return lastbool;
            }
        }
    }
    return lastbool;
}

////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Helper function for checking if a variable's pattern is already known
 *
 * @return true if already predicted
 */
////////////////////////////////////////////////////////////////////////////////////////////
bool
alreadyPredicted(char* function, char* variable) {
    int i = 0, k;
    //Find the function
    while((strcmp(function, functions[i].funcName) != 0)) {
        i++;
    }
    if(debug) printf("Found fxn\n");
    int varIndex = functions[i].varIndex;
    //Find the variable if it exists
    for(k = 0; k <= varIndex; k++) {
        if(strcmp(variable, functions[i].vars[k].varName) == 0) {
            if(debug) printf("Found var. Previous pattern: %s\n", functions[i].vars[k].predictable ? "Regular Access" : "Irregular Access");
            return functions[i].vars[k].predictable;
        }
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Qualifies the stride access pattern of a pointer based on its indices
 * If the variable has 2 indices, returns the higher stride index pattern
 *
 * SHORT < MEDIUM < LONG < VARIABLE < RANDOM
 *
 * @return The overall stride pattern of the pointer
 */
////////////////////////////////////////////////////////////////////////////////////////////
Stride
findStride(FILE* file, int fxnnum, int varnum) {
    struct Variable variable = functions[fxnnum].vars[varnum];
    Stride stride1 = SHORT, stride2 = SHORT, testStride;
    bool numericStride1, variableStride1, numericStride2, variableStride2;
    bool loopStride1, loopStride2;
    
    numericStride1 = false;
    numericStride2 = false;
    variableStride1 = false;
    variableStride2 = false;
    loopStride1 = false;
    loopStride2 = false;
    
    bool tidStride = false;

    int i;
    
    //Identify nature of first dependency
    for(i = 0; i < variable.instIndex; i++) {
        if(debug) printf("\t\tDep1: %s\n", variable.dependency[i]);
        if(strstr(variable.dependency[i], "%")) {
            if(strstr(variable.dependency[i], "indvar")) {
                loopStride1 = true;
            } else {
                variableStride1 = true;
            }
        } else if(strstr(variable.dependency[i], "tid")) {
	    tidStride = true;
	} else {
            numericStride1 = true;
        }
    }
    //Identify nature of second dependency if it exists
    if(variable.depNum == 2) {
        for(i = 0; i < variable.instIndex; i++) {
            if(debug) printf("\t\tDep2: %s\n", variable.dependency2[i]);
            if(strstr(variable.dependency2[i], "%")) {
                if(strstr(variable.dependency2[i], "indvar")) {
                    loopStride2 = true;
                } else {
                    variableStride2 = true;
                }
            } else {
		numericStride2 = true;
            }
        }
    }
    if(debug) printf("\t\t\tDep1 has variable indices: %s\n", variableStride1 ? "True" : "False");
    if(debug) printf("\t\t\tDep1 has loop indices: %s\n", loopStride1 ? "True" : "False");
    if(debug) printf("\t\t\tDep1 has numeric indices: %s\n", numericStride1 ? "True" : "False");

    //Perform Stride Qualification
    if(debug) printf("\t\t\tChecking Stride of Dep1\n");
    if(variableStride1 || (loopStride1 && numericStride1)) {
        if(debug) printf("\t\t\tPointer has variable index\n");
        return VARIABLE;
    } else if(tidStride) {
	stride1 = SHORT;
    } else if(loopStride1) {
        for(i = 0; i < variable.instIndex; i++) {
            if(debug) printf("\t\t\t\tPerforming Stride Traceback\n");
            rewind(file);
            stride1 = loopIndexStrideFinder(file, functions[fxnnum].funcName, variable.dependency[i]);
        }
    } else {
        stride1 = numericStrideFinder(variable, 1);
    }
    
    if(variable.depNum == 2) {
        if(debug) printf("\t\t\tDep2 has variable indices: %s\n", variableStride2 ? "True" : "False");
        if(debug) printf("\t\t\tDep2 has loop indices: %s\n", loopStride2 ? "True" : "False");
        if(debug) printf("\t\t\tDep2 has numeric indices: %s\n", numericStride2 ? "True" : "False");
        if(debug) printf("\t\t\tChecking Stride of Dep2\n");
        if(variableStride2 || (loopStride2 && numericStride2)) {
            if(debug) printf("\t\t\tPointer has variable index\n");
            return VARIABLE;
        } else if (loopStride2) {
            for(i = 0; i < variable.instIndex; i++) {
                rewind(file);
                stride2 = loopIndexStrideFinder(file, functions[fxnnum].funcName, variable.dependency2[i]);
            }
        } else {
            stride2 = numericStrideFinder(variable, 2);
        }
    } else {
        stride2 = SHORT;
    }
    
    return (stride1 > stride2) ? stride1 : stride2;
}

////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Qualifies the stride access pattern when all indices are numeric
 * Considers variable stride length to be MEDIUM if difference between max and min
 *     indices is small
 *
 * @return The overall stride pattern of numeric indices
 */
////////////////////////////////////////////////////////////////////////////////////////////
Stride
numericStrideFinder(struct Variable var, int depnum) {
    int min = 0, max = 0, stride = 0, laststride = 0, lastval = 0, i;
    char* dep;
    Stride pattern;
    for(i = 0; i < var.instIndex; i++) {
        if(depnum == 2) {
            dep = var.dependency2[i];
        } else {
            dep = var.dependency[i];
        }
        int val = atoi(strndup(strstr(dep, " ") + 2, strstr(dep, "\0") - strstr(dep, " ") + 1));

        if (i == 0) {
            min = val;
            max = val;
            lastval = val;
        } else if (i == 1) {
            laststride = val - lastval;
        }
        if(val > max) max = val;
        if(val < min) min = val;
        stride = val - lastval;
        if((i > 1) && (stride != laststride)) pattern = VARIABLE;
        if(debug) printf("\t\t\t%d,%d,%d,%d,%d,%d\n", min, max, val, lastval, stride, laststride);
        lastval = val;
        laststride = stride;
        
    }
    if(pattern == VARIABLE) {
        if((max - min) <= SHORTLIM) {
            return SHORT;
        } else if((max - min) <= 8) {
            return MEDIUM;
        } else {
            return VARIABLE;
        }
    }
    if(laststride <= SHORTLIM) {
        return SHORT;
    } else if(laststride <= MEDLIM) {
        return MEDIUM;
    } else {
        return LONG;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Qualifies the stride access pattern of a loop index by identifying the .next variable
 *     associated with the loop index and finding its stride length
 *
 * @return The stride pattern of the loop index
 */
////////////////////////////////////////////////////////////////////////////////////////////
Stride
loopIndexStrideFinder(FILE* file, char* fxnname, char* index) {
    char* rawindex;
    bool inFunction = false;
    
    if(strstr(index, " ")) {
        rawindex = strndup(strstr(index, " ") + 1, strstr(index, "\0") - strstr(index, " ") + 1);
    } else {
        rawindex = index;
    }
    
    char line[256];
    if(debug) printf("\tSearching for: %s\n", rawindex);
    if(strstr(rawindex, ".next")) {
        while (fgets(line, sizeof(line), file)) {
            if(!inFunction) {
                char* functionSearch = strstr(line, "efine ");
                if(functionSearch) {
                    if(debug) printf("\tFound a function! ");
                    char* argSearch = strstr(line, "(");
                    functionSearch += 6;
                    if(debug) printf("%s\n", strndup(functionSearch, argSearch - functionSearch));
                    if(strcmp(fxnname, strndup(functionSearch, argSearch - functionSearch)) == 0) {
                        inFunction = true;
                        if(debug) printf("\tFound %s\n", fxnname);
                    }
                }
            } else {
                char* findIndex = strstr(line, rawindex);
                if(findIndex && (*(findIndex + strlen(rawindex)) == ' ')) {
                    if(debug) printf("\t\tFound %s\n", rawindex);
                    char* findAdd = strstr(findIndex, "add");
                    if(!findAdd) {
                        return VARIABLE;
                    } else {
                        findIndex = strstr(findAdd, "indvars");
                        if(strstr(findIndex, "%")) {
                            return VARIABLE;
                        } else {
                            char* addedval = strstr(line, ", ") + 2;
                            int val = atoi(strndup(addedval, strstr(addedval, ",") - addedval));
                            
                            if(val <= SHORTLIM){
                                return SHORT;
                            } else if (val <= MEDLIM) {
                                return MEDIUM;
                            } else {
                                return LONG;
                            }
                        }
                    }
                }
            }
        }
    } else {
        while (fgets(line, sizeof(line), file)) {
            if(!inFunction) {
                char* functionSearch = strstr(line, "efine ");
                if(functionSearch) {
                    if(debug) printf("\tFound a function! ");
                    char* argSearch = strstr(line, "(");
                    functionSearch += 6;
                    if(debug) printf("%s\n", strndup(functionSearch, argSearch - functionSearch));
                    if(strcmp(fxnname, strndup(functionSearch, argSearch - functionSearch)) == 0) {
                        inFunction = true;
                        if(debug) printf("\tFound %s\n", fxnname);
                    }
                }
            } else {
                char* findIndex = strstr(line, rawindex);
                if(findIndex  && (*(findIndex + strlen(rawindex)) == ' ')) {
                    char* findphi = strstr(findIndex, "phi");
                    if(findphi) {
                        char* findNext = strstr(findphi, "%indvars.iv.next");
                        if(findNext) {
                            rewind(file);
                            return loopIndexStrideFinder(file, fxnname, strndup(findNext, strstr(findNext, ", ") - findNext));
                        }
                    }
                }
            }
        }
    }
}

void
printGlobals(FILE* outfile, int funcIndex, int varIndex) {
    int k;
    
    fprintf(outfile, "\n\tVariable: %s    ", functions[funcIndex].vars[varIndex].varName);
    Stride strde = functions[funcIndex].vars[varIndex].stride;
    if(strde < VARIABLE) {
        fprintf(outfile, "Prefetchable\n");
    } else {
        fprintf(outfile, "Not Prefetchable\n");
    }
    fprintf(outfile, "\tType: %s\n", functions[funcIndex].vars[varIndex].varType);
    for(k = 0; k < functions[funcIndex].vars[varIndex].instIndex; k++) {
        if(*functions[funcIndex].vars[varIndex].dependency2[k] != '\0') {
            fprintf(outfile, "\t\t%s -> %s, %s\n", functions[funcIndex].vars[varIndex].instruction[k],
                                      functions[funcIndex].vars[varIndex].dependency[k],
                                      functions[funcIndex].vars[varIndex].dependency2[k]);
        } else {
            fprintf(outfile, "\t\t%s -> %s\n", functions[funcIndex].vars[varIndex].instruction[k],
                                      functions[funcIndex].vars[varIndex].dependency[k]);
        }
    }
}

void
printLocals(FILE* outfile, int funcIndex, int varIndex) {
    int k;
    
    fprintf(outfile, "\n\tVariable: %s    ", functions[funcIndex].vars[varIndex].varName);
    Stride strde = functions[funcIndex].vars[varIndex].stride;
    if(strde < VARIABLE) {
        fprintf(outfile, "Prefetchable\n");
    } else {
        fprintf(outfile, "Not Prefetchable\n");
    }
    fprintf(outfile, "\tType: %s\n", functions[funcIndex].vars[varIndex].varType);
    for(k = 0; k < functions[funcIndex].vars[varIndex].instIndex; k++) {
        if(*functions[funcIndex].vars[varIndex].dependency2[k] != '\0') {
            fprintf(outfile, "\t\t%s -> %s, %s\n", functions[funcIndex].vars[varIndex].instruction[k],
                                      functions[funcIndex].vars[varIndex].dependency[k],
                                      functions[funcIndex].vars[varIndex].dependency2[k]);
        } else {
            fprintf(outfile, "\t\t%s -> %s\n", functions[funcIndex].vars[varIndex].instruction[k],
                                      functions[funcIndex].vars[varIndex].dependency[k]);
        }
    }
}
