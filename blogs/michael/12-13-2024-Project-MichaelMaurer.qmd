# EECE 7309: Final Project
# Register Allocator

## Code Availability
The code is available at https://github.com/maurermi/compilers-final-project

To run the register allocator on a given Bril file, run the following:

```{bash}
python relabel.py {filename}.bril > {filename}-relabeled.bril
bril2json < {filename}-relabeled.bril > {filename}.json
python cfg.py {filename}.json
```

To generate results, run 
```{bash}
./test-all.sh
```

## Introduction

This project provides code which converts Bril programs to SSA, computes live variables by block, and then allocates registers to all instructions in the program. Two register allocation algorithms are implemented, a linear scan allocator, and a graph-coloring based allocator. Additionally, I developed a working backend for the RISC-V architecture, and can generate code which runs on a RISC-V simulator.

This allocator is designed specifically for a RISC-V based system, namely RV32E (which has 32 32-bit register).

The allocator here generates code which can be run directly in RISC-V, and was tested on this simulator: https://ascslab.org/research/briscv/simulator/simulator.html.

One goal of this project was to compare head to head the time incurred by the two algorithms. Namely, I was interested in benchmarking the time elapsed by the graph coloring allocator as opposed to the linear scan allocator. 

Another goal was to implement a working backend for the RISC-V architecture, and to generate code which runs on a RISC-V simulator.

### Preface

A large reason I chose this course was because I have for a long time been interested in learning more about compilers, but namely because compilers were often a black box to me. In terms of the pipeline which goes from code to electrons, compilers were really the only part I had not yet explored. 
I chose a register allocator as this project because I feel that it completes the chain, and now I feel as though I can connect implementation ideas all the way through the chain, from high level language design to the hardware. 

When proposing this project, I actually had a seperate project in mind, which I worked on for a while, but realized that there was a very slim chance I complete it before the end of this semester. 
This was a transpiler of high level language into Circom, a language which defines a set of constraints for a proof. 
This project is still ongoing, however in a seperate capacity these days. When I realized that this wouldn't be feasible for this project, I switched to this register allocator project as an alternative. 

## Goals

The main goal of this project is to gain hands-on familiarity with the algorithms used in compiler implementation, as well as to explore some of the topics discussed in class, not previously covered in assignments.
As such, I felt that using Bril would be more educational than LLVM, as it required me to implement essentially everything from scratch. In a follow on work, I plan to do the same in LLVM, because it would be nice to have it around. 

## Challenges

One unexpected challenge during this work was how tricky it was to convert to SSA. Adrian Sampson, in his lecture on SSA, does actually mention that converting to SSA is trickier than one may expect, and I can confirm that this is the case. Conceptually, it is not terribly challenging, but while implementing it, I developed a massive chunk of code, that was difficult to effectively partition, and this made it especially difficult to debug. Namely, the part which was challenging was not adding in phi nodes, but rather renaming variables properly. To me, it does not seem as though this was a conceptual issue, more of a software engineering one. The amount of code I wrote to support SSA is a significant chunk of this code, and got to the point at which I was spending more time debugging the SSA conversion than the register allocation, which is not what I was aiming for. 
So, that being said, the SSA conversion code in this project is not fully working, because of the rename step. However, I did write a significant amount of infrastructure to perform register allocation once SSA form is achieved. 

Another challenge in this project was determining whether I should do this work using LLVM or Bril. In retrospect, it would have been better to use LLVM as the main reason I wanted to use Bril was to force myself to better understand the algorithms at play here. However, given that the main advantage LLVM would have given is SSA formatted code, it may have been wiser (and admittedly, I'm more comfortable with C++ than Python). However given what I learned during this project, I have no concerns about taking this work and applying it to an LLVM based register allocator. 

One lesson is that I should get better at writing unit tests.

## Implementation

### Register Allocation Algorithms
There are two register allocation algorithms implemented here, a linear scan allocator, and a graph coloring allocator. 
Linear scan is a simpler algorithm, which is generally faster, but does not always perform as well as graph coloring. 
Previously, I have read and implemented a similar algorithm for scheduling tasks in a cloud computing environment, based on a paper from Northeastern's Xue Lin. 

The basis of the linear scan algorithm is essentially two phases:
1. Computation of the interference graph
2. Register Assignment

In the first phase, one must compute the liveness of variables by block, and determine which variables are live at the same time as one another. 
This analyisis is very similar to some of the data flow analyses we have done in class. Essentially, a worklist of basic blocks is created, in reverse post-order. Then, for each block we encounter, we compute the live out variables, which are the union of live in variables in all successors. Then, we compute the live in variables, which are the union of used variables, and the set difference between live out variables and defined variables in the block. In my implementation, I included some timing information for this analysis.

```{python}
    def compute_liveness(self):
        import time
        # Initialize the blocks
        for block in self.blocks.values():
            block.compute_used_and_defined_vars()
            block.live_out = set()
            block.live_in = set()
        # Create worklist
        worklist = self.terminating_blocks
        start_time = time.time()
        while True:
            # Continue until no changes are made
            changed = False
            # This prevents infinite loops
            blocks_seen = set()
            while len(worklist) > 0:
                # Work through blocks in reverse-post order
                block = self.blocks[worklist.pop()]
                old_live_in = copy.deepcopy(block.live_in)
                old_live_out = copy.deepcopy(block.live_out)
                blocks_seen.add(block.name())
                # Compute live out variables
                for succ in block.succs:
                    block.live_out = block.live_out.union(self.blocks[succ].live_in)
                # Compute live in variables
                block.live_in = block.used_vars.union(block.live_out - block.defined_vars)
                # Check if changes were made
                if old_live_in != block.live_in or old_live_out != block.live_out:
                    changed = True
                # Add predecessors to worklist
                for pred in block.preds:
                    # This prevents infinite loops
                    if pred not in blocks_seen:
                        worklist.add(pred)
            if not changed:
                break
        end_time = time.time() 
        print("Computing Liveness Time taken: ", end_time - start_time)
        return
```

The next step is register allocation. In the linear scan algorithm, we iterate through instructions and assign registers to the destinations. Further, at each instruction, we scan the allocated registers to see if any can be freed (whether the given instruction is past the last use). Allocation is done by choosing the first free register if there is one, and assigning a value to it. If there are none available, there must be a spill. Spilling is typically done based on certain heuristics. Here, I experimented with two heuristics, one being random choice, and the other being choosing the register which has the latest "last use". Apparently, this is a popular heuristic. 

A shortcoming of the naive linear scan register allocation strategy is that it does do much in the way of wisely allocating registers. Namely, it's possible that fewer registers are required than what the allocation scheme calls for. So if register pressure is a major concern, linear scan register allocation is not a great choice. However, it is reported to be quite a bit faster than many of the existing ahead of time register allocation strategies (namely graph coloring). 

One worthwhile note is that the linear scan register allocator requires the ability to iterate through blocks in order. My CFG class however uses a dictionary for storing blocks, which hashes entries, so either when initializing the CFG I would need to tag instructions with their order in the program, or I could preprocess the program to add labels before certain instructions. I ended up choosing the latter, so there is a tool with this work called `relabel.py` aimed at renaming labels by the number instruction they are in the program. 

The second register allocation strategy implemented was a graph coloring algorithm (Chiatin's algorithm). 
This is a fairly famous technique, which we discussed in class and is also discussed heavily in the literature. The idea here is to treat the program as a graph, with each variable as a vertex. Then, each edge connects two verticies which are alive at the same time. 

The first step in this scheme is therefore, to create the interference graph. This work was inherited directly from the linear scan register allocation. 

Next, a copy of the graph is made, and is "simplified". Simplificaiton refers to an iteritave process by which nodes and edges are removed from the graph. Verticies are removed by determining, first, whether there are any verticies that exist in the graph which have fewer than `k` edges, where `k` is the number of registers available to the program. If so, the first (in my implementation) vertex found meeting this criterion is removed from the graph, and it along with its outgoing edges are pushed onto a stack, with a flag indicating that this vertex is to be "colored", or to have a register allocated to it. If no vertex exists meeting this criterion, then a vertex is chosen at random. This vertex and its edges are removed, and pushed onto the stack with a flag indicating that it should _not_ be colored. This is continued until the fixed point when there are no remaining verticies in the graph copy, is reached. 

Next, verticies are added back into the graph, by popping them from the aforementioned stack. If the vertex popped is marked to be colored, it is assigned a color (register) which is different from any verticies with which it shares an edge. If it is marked to not be colored, first, it is checked whether there are any free registers (not in use by neighbors). If so, this register is assigned, and the variable does not actually need to be spilled. If not, then the variable is marked to be spilled to memory. If it is spilled, then it is not added back into the graph.
If spills occur, this means that loads and stores must be inserted into the resultant code. This is handled in the actual register assignment code. 

One implementation note, is that these algorithms are implemented to operate on a set number of registers, but are not yet provided the actual register IDs. This is because I wanted these algorithms to be machine independent. In the next step, these register tags are converted to true physical registers, following the requirements of a given architecture.

Another quirk is that the register allocation is done on a per-function basis. So there is no cross-function register allocation optimization done in this process. This would make the analysis much tricker (namely because of the control flow changes), but is an interesting avenue of future work.

### Register Assignment on Machine
For this project, I focused on a RISC-V architecture, and this code, with some exceptions, can develop code which runs on a RISC-V simulator. This was not really part of my initial plan, but I found it quite exciting to be able to generate code that can actually run, and do what it claims to on a target machine. 

There are some exceptions, namely operations which are not supported natively by RISC-V (or by the limited instruction set provided by RV32I). For some operations (such as Bril's `br` operation), I added support through the use of a mixture of non-equivalent operations. These were some common instructions used that have natural analog constructions in RISC-V. One example would be the `eq` instruction. This has no analog in RV32I, but an equivalent can be achieved with a subtraction and logical inverse. This is also commonly seen in tandem with `br`, so a branch based on the result of an `eq` can be replaced by a `sub` operation followed by `beq _ zero` in RISC-V. Some operations have no real equivalent, such as `print`, so those are translated directly for the time being. 

For the code generated, I reccomend giving it a try on BU's RISC-V simulator. For workloads containing only instructions supported by RV32I, the code should run and produce results!

I used the following register constraints in my setup:
Argument Registers = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
Return Register = ['ra']
General Purpose Registers = ['t0', 't1', 't2', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']
Special Registers = ['fp', 'sp', 'gp', 'tp', 'zero']

The backend really only conisders the argument registers, return registers, and general purpose registers. The stack pointer is used for inserting load/store instructions, but otherwise is left alone. 

## Evaluation

I used SIGPLAN to help develop reasonable evaluation workloads for this project. For sufficient breadth of testing and number of trials, I evaluated this work on a large portion of Bril examples provided in the Bril repository (under benchmarks/core) totaling 35 different workloads. Those used are provided alongside this code. 

There are two forms of evaluation used hereafter. One is determining whether the code used assigns registers legally (i.e. does not double-assign a register, spills when appropriate). 
The other evaluation is a head to head comparison of the register assignment techniques used. 
Here, we compare the number of registers used by the assignment strategy (at maximum), and the number of spills for a given workload and number of available registers. 
Additionally, we measure the time taken to complete the register assignment operation. 
For graph coloring, I expect it to be on average slower than linear-scan, but I expect graph-coloring on average to have fewer spills and use fewer registers. 

** Note: I chose to manually pick the number of available registers rather than giving the system default to stress test the system. Further, in multithreaded applications this behavior could be useful. 

## Results

### Performance Results
Here, we validate the perofrmance of the register allocator on systems with 5, 7, and 10 available genereal purpose registers. 

First, we confirm that all test cases produce valid register assignments. Using the provided test script (`test-all.sh`), we can confirm that all tests pass, and generate a set of test results to operate over. 

Next, we investigate performance differences between the linear scan register allocation and graph-coloring based register allocation. 

First, the top line numbers are as follows:
- Linear Scan register allocation is on average, 9.75x faster than graph coloring
- Graph coloring on average uses 84.4% of the registers used by linear scan
- Graph coloring on average has 0.141 fewer spills than linear scan allocation

These results tend to align with what I expected. 
Investigating the results further, it appears that generally, the linear scan-based register allocator and graph-coloring based allocator have a similar number of spills (often the same).
The number of registers allocated by the linear-scan register allocator is often larger than the number of registers allocated by the graph-based allocator. 
Now most notably, the linear scan register allocator is nearly 10x as fast as the graph colloring allocator on average. This is clear evidence for why it is typically used in JIT systems, whereas graph coloring is really only used in ahead of time compilation.

Another note is that randomly spilling registers (when a spill is required) appears to perform slightly worse than spilling the register that has the longest time to be free. The difference is not large however (on average, 0.02 more spills with random spillage).

### RISC-V Code Generation
The other chunk of this program was implementing a backend to convert from bril to RISC-V. Admittedly, this addition was somewhat of an afterthought, and I was not able to automate the test infrastructure to determine correctness of these RISC-V programs. Additionally, the backend does not support all Bril instructions, so much of the testing here was done by hand on curated examples. 

In the future, I am hoping to further flesh this out, because I think this was quite fun and interesting. In hindsight, I would have preferred to focus more of my project on this aspect. 

Sample RISC-V Program:
```{assembly}
main:
.l0:
	li t1, 10
.l1:
	li t0, 1
.l2:
	li t2, 1
.l3:
	add t0, t1, t0
.l4:
	li s1, 50
.l5:
	beq s1, zero, .l6
	j l7
.l6:
	j .l11
.l7:
	add t0, t0, t2
.l11:
	add t0, t2, t0
.l8:
	beq s1, zero, .l9
	j l10
.l9:
	j .l11
.l10:
	mul t0, t0, t2
.l12:
	add t2, t2, t2
.l13:
	add t0, t0, t2
.l14:
	add t0, t0, t1
.l15:
	mv a0, t0
	jr ra
```

Corresponding Bril:
```{bril}
@main() {
.l0:
  n: int = const 10;
.l1:
  one: int = const 1;
.l2:
  a: int = const 1;
.l3:
  b: int = add n one;
.l4:
  magic: int = const 50;
.l5:
  branch: bool = gt b a;
  br branch .l6 .l7;
.l6:
  b = id a;
  jmp .l11;
.l7:
  b = add b a;
.l8:
  branch: bool = gt b magic;
  br branch .l9 .l10;
.l9:
  b = sub b a;
  jmp .l11;
.l10:
  b = mul b a;
# There should be a 3-way phi node here.;
.l11:
  c: int = add a b;
.l12:
  d: int = add a a;
.l13:
  e: int = add c d;
.l14:
  f: int = add e n;
.l15:
  ret f;
}
```

## Conclusion

This project implements two register allocation techniques, compares the results, and further creates a limited scope backend which generates actual, runnable RISC-V code. 
The results of this were as expected and line up with the literature, in that linear scan register allocation is much faster than graph-coloring based allocation, but generally speaking it performs worse. I found it interesting to verify this, because although it's clear algorithmically, it's nice to see that on actual workloads the behavior holds. 
I learned quite a bit throughout this project, and am glad I focused on a register allocator. Now I feel as though I have a good sense for every step going from high level code to hardware. Additionally, this work prompted some exploration of graph theory on my end, which I would like to pursue more. 

## Future Work

I would like to perform this exercise on LLVM instead of Bril in the future. I think leveraging the existing tooling would have allowed me to iterate much faster on this work, and possibly come up with more interesting results. However, I think that using Bril as the basis here really allowed me to get a good understanding of the algorithms and techniques at play. 

It would be interesting to expand the backend to support additional architectures, namely ones with more complex register behavior.
One example would be x86. Additionally, adding more instruction support would be very helpful, as then arbitrary bril programs could be converted to RISC-V using this tool.

## References
1. https://ieeexplore.ieee.org/abstract/document/6985666
1. https://www.youtube.com/watch?v=eeXk_ec1n6g&t=1039s
1. https://www.youtube.com/watch?v=eWp_-XCwN1A
1. https://www.cs.cornell.edu/courses/cs6120/2023fa/lesson/6/ 
2. https://www.cs.cornell.edu/courses/cs6120/2023fa/lesson/4/ 
3. https://capra.cs.cornell.edu/bril/lang/ssa.html
4. https://normrubin.github.io/lectures/04_data_flow.html
5. https://normrubin.github.io/lectures/register_allocation.html 
6. https://anoopsarkar.github.io/compilers-class/assets/lectures/opt3-regalloc-linearscan.pdf 
7. https://homepages.dcc.ufmg.br/~fernando/classes/dcc888/ementa/Questions/RegisterAllocation0.pdf 
8. https://compilers.cs.uni-saarland.de/papers/ssara.pdf 
9. https://en.wikipedia.org/wiki/Register_allocation 
10. https://web.stanford.edu/class/archive/cs/cs143/cs143.1128/lectures/17/Slides17.pdf 
11. https://github.com/riscv/riscv-isa-manual/blob/main/src/rv32.adoc 

## Appendix

### Sample Graph Coloring Register Allocation Visualization
```{mermaid}
graph TD    
    magic:0 --> n:1
    d:0 --> n:1
    e:0 --> n:1
    branch:2 --> n:1
    branch:2 --> magic:0
    b:3 --> n:1
    magic:0 --> b:3
    branch:2 --> b:3
    a:4 --> n:1
    b:3 --> a:4
    magic:0 --> a:4
    branch:2 --> a:4
    c:2 --> n:1
    c:2 --> a:4
    d:0 --> c:2
    one:0 --> n:1
    a:4 --> one:0
```

### Sample Linear Scan Register Allocation Visualization
```
# tests/sample.bril
0: n		1:  		2:  		3:  		4:  		5:  		6:  		
0: n		1: one		2:  		3:  		4:  		5:  		6:  		
0: n		1: one		2: a		3:  		4:  		5:  		6:  		
0: n		1:  		2: a		3: b		4:  		5:  		6:  		
0: n		1: magic	2: a		3: b		4:  		5:  		6:  		
0: n		1: magic	2: a		3: b		4: branch	5:  		6:  		
0: n		1: magic	2: a		3: b		4: branch	5:  		6:  		
0: n		1: magic	2: a		3: b		4: branch	5:  		6:  		
0: n		1: magic	2: a		3: b		4: branch	5:  		6:  		
0: n		1: magic	2: a		3: b		4: branch	5:  		6:  		
0: n		1:  		2: a		3: b		4: branch	5: c		6:  		
0: n		1: d		2: a		3: b		4:  		5: c		6:  		
0: n		1: d		2: a		3: b		4: e		5: c		6:  		
0: n		1: d		2: a		3: b		4: e		5: c		6: f		
0: n		1: d		2: a		3: b		4: e		5: c		6: f		
0: n		1: d		2: a		3:  		4: e		5: c		6: f
```

### Sample Test Result Data
```
Head to head comparison
Name: ackermann_ack_10 Graph Execution Time:  0.00449681282043457 Linear Execution Time:  0.0001437664031982422 Speedup:  31.27860696517413
Name: ackermann_ack_7 Graph Execution Time:  0.0022280216217041016 Linear Execution Time:  0.0002560615539550781 Speedup:  8.701117318435754
Name: ackermann_ack_5 Graph Execution Time:  0.002377033233642578 Linear Execution Time:  0.00023508071899414062 Speedup:  10.111561866125761
Name: ackermann_main_10 Graph Execution Time:  9.799003601074219e-05 Linear Execution Time:  3.0040740966796875e-05 Speedup:  3.261904761904762
Name: ackermann_main_7 Graph Execution Time:  8.177757263183594e-05 Linear Execution Time:  2.288818359375e-05 Speedup:  3.5729166666666665
Name: ackermann_main_5 Graph Execution Time:  7.414817810058594e-05 Linear Execution Time:  1.9788742065429688e-05 Speedup:  3.746987951807229
Name: binary_main_10 Graph Execution Time:  5.412101745605469e-05 Linear Execution Time:  2.09808349609375e-05 Speedup:  2.5795454545454546
Name: binary_main_7 Graph Execution Time:  4.506111145019531e-05 Linear Execution Time:  5.91278076171875e-05 Speedup:  0.7620967741935484
Name: binary_main_5 Graph Execution Time:  3.409385681152344e-05 Linear Execution Time:  1.3113021850585938e-05 Speedup:  2.6
Name: binary_printBinary_10 Graph Execution Time:  0.0008351802825927734 Linear Execution Time:  6.413459777832031e-05 Speedup:  13.022304832713754
Name: binary_printBinary_7 Graph Execution Time:  0.0008020401000976562 Linear Execution Time:  0.00011515617370605469 Speedup:  6.9648033126294
Name: binary_printBinary_5 Graph Execution Time:  0.0007398128509521484 Linear Execution Time:  9.799003601074219e-05 Speedup:  7.549878345498784
...
Name: totient_totient_10 Graph Execution Time:  0.0045719146728515625 Linear Execution Time:  0.0003387928009033203 Speedup:  13.494722026741732
Name: totient_totient_7 Graph Execution Time:  0.00251007080078125 Linear Execution Time:  0.000286102294921875 Speedup:  8.773333333333333
Name: totient_totient_5 Graph Execution Time:  0.0029671192169189453 Linear Execution Time:  0.00013017654418945312 Speedup:  22.793040293040292
Name: totient_mod_10 Graph Execution Time:  0.0002512931823730469 Linear Execution Time:  4.100799560546875e-05 Speedup:  6.127906976744186
Name: totient_mod_7 Graph Execution Time:  0.0002460479736328125 Linear Execution Time:  3.409385681152344e-05 Speedup:  7.216783216783217
Name: totient_mod_5 Graph Execution Time:  0.0002429485321044922 Linear Execution Time:  3.0994415283203125e-05 Speedup:  7.838461538461538
Name: up_main_10 Graph Execution Time:  0.0001239776611328125 Linear Execution Time:  3.409385681152344e-05 Speedup:  3.6363636363636362
Name: up_main_7 Graph Execution Time:  0.00014209747314453125 Linear Execution Time:  2.7179718017578125e-05 Speedup:  5.228070175438597
Name: up_main_5 Graph Execution Time:  0.00010538101196289062 Linear Execution Time:  2.4080276489257812e-05 Speedup:  4.376237623762377
Name: up_up_arrow_10 Graph Execution Time:  0.0010821819305419922 Linear Execution Time:  0.00011086463928222656 Speedup:  9.761290322580646
Name: up_up_arrow_7 Graph Execution Time:  0.0010287761688232422 Linear Execution Time:  8.916854858398438e-05 Speedup:  11.537433155080214
Name: up_up_arrow_5 Graph Execution Time:  0.0010311603546142578 Linear Execution Time:  7.987022399902344e-05 Speedup:  12.91044776119403
Average speedup of Linear:  9.752766747717136x
Name: ackermann_ack_10 Graph Registers:  4 Linear Registers:  6 Ratio:  0.6666666666666666
Name: ackermann_ack_7 Graph Registers:  4 Linear Registers:  6 Ratio:  0.6666666666666666
Name: ackermann_ack_5 Graph Registers:  4 Linear Registers:  5 Ratio:  0.8
Name: ackermann_main_10 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: ackermann_main_7 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: ackermann_main_5 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: binary_main_10 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: binary_main_7 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: binary_main_5 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: binary_printBinary_10 Graph Registers:  3 Linear Registers:  3 Ratio:  1.0
Name: binary_printBinary_7 Graph Registers:  3 Linear Registers:  3 Ratio:  1.0
Name: binary_printBinary_5 Graph Registers:  3 Linear Registers:  3 Ratio:  1.0
...
Name: totient_main_10 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: totient_main_7 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: totient_main_5 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: totient_totient_10 Graph Registers:  6 Linear Registers:  7 Ratio:  0.8571428571428571
Name: totient_totient_7 Graph Registers:  6 Linear Registers:  7 Ratio:  0.8571428571428571
Name: totient_totient_5 Graph Registers:  5 Linear Registers:  5 Ratio:  1.0
Name: totient_mod_10 Graph Registers:  2 Linear Registers:  3 Ratio:  0.6666666666666666
Name: totient_mod_7 Graph Registers:  2 Linear Registers:  3 Ratio:  0.6666666666666666
Name: totient_mod_5 Graph Registers:  2 Linear Registers:  3 Ratio:  0.6666666666666666
Name: up_main_10 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: up_main_7 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: up_main_5 Graph Registers:  1 Linear Registers:  1 Ratio:  1.0
Name: up_up_arrow_10 Graph Registers:  4 Linear Registers:  6 Ratio:  0.6666666666666666
Name: up_up_arrow_7 Graph Registers:  4 Linear Registers:  6 Ratio:  0.6666666666666666
Name: up_up_arrow_5 Graph Registers:  4 Linear Registers:  5 Ratio:  0.8
Average Registers used by Graph vs Linear:  0.8438802083333327x
Name: ackermann_ack_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: ackermann_ack_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: ackermann_ack_5 Graph Spills:  0 Linear Spills:  1 Difference:  1
Name: ackermann_main_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: ackermann_main_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: ackermann_main_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_main_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_main_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_main_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_printBinary_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_printBinary_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: binary_printBinary_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
...
Name: totient_main_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_main_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_main_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_totient_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_totient_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_totient_5 Graph Spills:  1 Linear Spills:  1 Difference:  0
Name: totient_mod_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_mod_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: totient_mod_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_main_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_main_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_main_5 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_up_arrow_10 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_up_arrow_7 Graph Spills:  0 Linear Spills:  0 Difference:  0
Name: up_up_arrow_5 Graph Spills:  0 Linear Spills:  1 Difference:  1
Average Spills used by Linear vs Graph:  0.140625
```