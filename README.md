## PRNG with rule 30

This is my final presentation of the course of Advanced Algorithm. It is about **Pseudo Random Number Generator (PRNG)** and **Celluear Automation (CA)**.

### 1. Structure

There are totally 4 directories, i.e, **NIST** for NIST randomness test, **PRNG** for different PRNGs including *Rule 30*, **reference** for somthing to refer and **utils** for some tools.

### 2. Codes

#### 2.1 NIST

This is a directory of NIST relative codes.

##### 2.1.1 Draft version of all tests

**NIST_test.py** is a draft version of NIST randomness test, including all 15 tests. You might find it difficult to read for some test functions cause there is no or little comment.

##### 2.1.2 Package Tests (Recommended, Remain to finish)

This is a package consist of all 15 **tets modules** with better comments. I would suggest you to refer to this package so that you could know about NIST randomness test quickly.

#### 2.2 PRNG

##### 2.2.1 PRNG (Remain to finish)

This file includes some differnt PRNG algorithms. 

##### 2.2.2 rule30

The code contains the functions of creating a **rule30 CA** (*ruel30*), **PRNG** with rule30 (*PRNG*) and **draw the picture** of rule30 (*draw_pic*).

#### 2.3 utils

This is a directory of some utils functions, no matter it is related to NIST or not.

##### 2.3.1 get_gif

To change the pictures saved before in the `rule30.draw_pic` function into the **gif** picture. Certainly, you could use it to change whatever a serials of images into gif.

##### 2.3.2 NIST_utils

There are many functions relative with NIST tests, e.g. `normalize` to change the 01 strings into {1,-1}, `string2list` to change the 01 strings into list.

### 3. Reference

#### 3.1 NIST User Manul

**A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications** explains all the randomness test, including implenmations and priciples. 

 ### 3.2 Radom Sequence

Writen by **Stephen Wolfram**, the paper explains random sequence generation by cellular automata.



##TO DO

-[ ] Complete the test modules

-[ ] Implement other CA rules

