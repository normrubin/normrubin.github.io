---format:
  html: default
tbl-colwidths:
- 10
- 20
- 20
- 20
- 15
- 15
title: EECS7398 Weekly Schedule
---


tbl-colwidths: [10,20,20,20, 15, 15 ]

Since the is the first time this course is offered.
This is a tentative schedule.

::: {.callout-warning}
Need to figure out where discussions are kept. Is it on github? or in canvas?
:::

|week  | day     | Date    | topic  | discussions | Due |
|------|---------|---------|--------|----         | ----|
| 1    |Friday|Sept  6|[Compiler overview and structure](./lectures/010_compiler_overview.qmd) |

|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
|2     | Tuesday | Sept 10 |[Performance Measurement](./lectures/01a_performance_measurement.qmd)|| [hw0](homework/hw0)


paper 1 - [Producing Wrong Data Without Doing Anything Obviously Wrong!](https://dl.acm.org/citation.cfm?id=1508275)
Todd Mytkowicz, Amer Diwan, Matthias Hauswirth, and Peter F. Sweeney. ASPLOS 2009.

LEADER: Norm using Adrian's blog)

[SIGPLAN Empirical Evaluation Guidelines](https://www.sigplan.org/Resources/EmpiricalEvaluation/)



|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
|2  | Friday  | Sept 13  |[Representing programs](./lectures/02a_representation.qmd)| |



|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
|3   | Tuesday | Sept 17 | [Overview of Bril](./lectures/02b_bril.qmd)  |

|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
 3   | Friday  | Sept 20 | [Local analysis and optimization](./lectures/03_local.qmd)  

|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
 4  | Tuesday | Sept 24    | [Value numbering](./lectures 03b_local_value_numbering/qmd) | |[hw1](homework/1.hw.qmd)

|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
 4   | Friday  | Sept 27 |  [Data flow](./lectures/04_data_flow.qmd)


|      |         |         |        |             |     |
|------|---------|---------|--------|----         | ----|
 5  | Tuesday | Oct 1 | [Global analysis](./lectures/05_global.qmd)|| [hw2](homework/2_hw.qmd)

 |      |         |         |       |             |     |
|------|---------|---------|--------|----         | ----|
 5   | Friday  | Oct 4 |   [loop invariant code motion](./lectures/05b_licm.qmd)


paper 2 - [iterative data-flow analysis, revisited](https://repository.rice.edu/server/api/core/bitstreams/790ce776-44cf-4474-8f60-4c1f5959ee74/content)
Cooper, Keith D.; Harvey, Timothy J.; Kennedy, Ken (2004-03-26) [November 2002]. pldi 2002
  
  
 |      |         |         |       |             |     |
|------|---------|---------|--------|----         | ----|
 6  | Tuesday | Oct 8  |[Static single assignment](./lectures/06_ssa.qmd)
 
Dynamo: A Transparent Dynamic Optimization System Vasanth Bala vas@hpl.hp.com 
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://dl.acm.org/doi/pdf/10.1145/349299.349303


 |      |         |         |       |             |     |
|------|---------|---------|--------|----         | ----|
 6  | Friday  | Oct 11 | continued 
 Global value numbers and redundant computations
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://dl.acm.org/doi/pdf/10.1145/73560.73562

 |      |         |         |       |             |     |
|------|---------|---------|--------|----         | ----|
 |7   | Tuesday | Oct 15 |[LLVM](./lectures/07_llvm.ipynb)


 https://dl.acm.org/doi/10.1145/1064978.1065042 Threads cannot be implemented as a library
 
 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |7   | Friday | Oct 18 | continued

 final project propsal ????

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |8| Tuesday | Oct 22 |  [Classical loop optimizations](./lectures/)

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |8   | Friday | Oct 25 |[Polyhedral analysis](./lectures/09_poly.qmd)

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |9   | Tuesday | Oct 29  | continued

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |9   | Friday | Nov 1 | [MLIR](./lectures/100_mlir.qmd)

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |10   | Tuesday | Nov 5 | continued  | |

 Superoptimizer: A Look at the Smallest Program
Alexia Massalin. ASPLOS 1987.

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |10  | Friday | Nov 8 | [Interprocedural Analysis](./lectures/110_whole_program.qmd)

 Formal Verification of a Realistic Compiler
Xavier Leroy. CACM in 2009.


 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
11   | Tuesday | Nov 12  | continued  | |

Efficient Path Profiling
Thomas Ball and James R. Larus. MICRO 1996.

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |11  | Friday | Nov 15  | [Memory Management](./lectures/12_memory.qmd)
 An Efficient Implementation of SELF, a Dynamically-Typed Object-Oriented Language Based on Prototypes
C. Chambers, D. Ungar, and E. Lee. OOPSLA 1989.


 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |12   | Tuesday | Nov 19 | continued  | |

 "Partial Redundancy Elimination" by Jens Knoop, Oliver Rüthing, and Bernhard Steffen

Year: 1992

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |12   | Friday | Nov 22  |  [Dynamic compilers](./lectures/13_dynamic_compilers.qmd)

 Formal Verification of a Realistic Compiler
Xavier Leroy. CACM in 2009.

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |13  | Tuesday | Nov 26  | continued  | |

 

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |13  | Friday | Nov 29 | **Thanksgiving**  | |

  https://dada.cs.washington.edu/research/tr/2017/12/UW-CSE-17-12-01.pdf
 12-01 TVM:End-to-End Optimization Stack for Deep Learnin

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |14  | Tuesday | Dec 3 |  [GPU Compilers](./lectures/14_gpu_compilers.qmd)

 Revealing Compiler Heuristics through Automated Discovery and Optimization, V. Seeker, C. Cummins, M. Cole, B. Franke, K. Hazelwood, H. Leather

 |      |         |         |        |            |     |
|------|---------|---------|--------|----         | ----|
 |14  | Friday | Dec 6 | continued  | |

 End-to-end deep learning of optimization heuristics - Chris Cummins, Pavlos Petoumenos, Zheng Wang, and Hugh Leather PACT 2017.
 https://ieeexplore.ieee.org/document/8091247

final project deadline ------ 
        
https://ieeexplore.ieee.org/document/10444819
A. Murtovi, G. Georgakoudis, K. Parasyris, C. Liao, I. Laguna and B. Steffen, "Enhancing Performance Through Control-Flow Unmerging and Loop Unrolling on GPUs," 2024 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), Edinburgh, United Kingdom, 2024, pp. 106-118, doi: 10.1109/CGO57630.2024.10444819. keywords: {Codes;Costs;Graphics processing units;Prototypes;Benchmark testing;Predictive models;Optimization;compiler;code duplication;LLVM;GPU},


https://escholarship.org/uc/item/3rt0n0q2
Gal, A., Probst, C. W, & Franz, M. (2003). A denial of service attack on the Java bytecode verifier. UC Irvine: Donald Bren School of Information and Computer Sciences. Retrieved from https://escholarship.org/uc/item/3rt0n0q2


https://dl.acm.org/doi/pdf/10.1145/3620665.3640366
PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation ASPLOS '24: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2


https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/448997590_1496256481254967_2304975057370160015_n.pdf?_nc_cat=106&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=4Yn8V9DFdbsQ7kNvgEwOdGk&_nc_ht=scontent-bos5-1.xx&oh=00_AYD-0YTCXuS11WU8rqC3N2aA-AfiflOptch_BD__V1V3xA&oe=6684630D

Meta Large Language Model Compiler: Foundation Models of Compiler Optimization Chris Cummins†, Volker Seeker†, Dejan Grubisic, Baptiste Rozière, Jonas Gehring, Gabriel Synnaeve, Hugh Leather†



---- papers 

1987 
[Superoptimizer: A Look at the Smallest Program](https://dl.acm.org/doi/abs/10.1145/36177.36194)
Alexia Massalin. ASPLOS 1987.


1988 
[Global value numbers and redundant computations](/https://dl.acm.org/doi/pdf/10.1145/73560.73562)
Rosen, B.K., Wegman, M.N. and Zadeck, F.K., popl 1988

2000

[Dynamo: A Transparent Dynamic Optimization System](/https://dl.acm.org/doi/pdf/10.1145/349299.349303)
Bala, V., Duesterwald, E. and Banerjia, S., PLDI 2000 


2002 
[iterative data-flow analysis, revisited](https://repository.rice.edu/server/api/core/bitstreams/790ce776-44cf-4474-8f60-4c1f5959ee74/content)
Cooper, Keith D.; Harvey, Timothy J.; Kennedy, Ken (2004-03-26),November 2002


2005
[Threads cannot be implemented as a library] (https://dl.acm.org/doi/10.1145/1064978.1065042)
 Boehm, H.J.. PLDI 2005


2015 
[Provably correct peephole optimizations with alive](https://dl.acm.org/doi/pdf/10.1145/2737924.2737965?casa_token=o9UQe90sRVwAAAAA:thVHM1EjwKgubb_CO07_pqFVz2SZFbkGiaPxUsMdMv5DZqFVqNJoTIXTZ1MwbCYZSm0i-49M_eqY)
Lopes, N.P., Menendez, D., Nagarakatte, S. and Regehr, J. pldi 2015

2009
[Formal Verification of a Realistic Compiler](https://dl.acm.org/doi/pdf/10.1145/1538788.1538814)
Xavier Leroy. CACM  2009.


2018
[TVM: end-to-end optimization stack for deep learning](https://dada.cs.washington.edu/research/tr/2017/12/UW-CSE-17-12-01.pdf)
 Chen, Tianqi, Thierry Moreau, Ziheng Jiang, Haichen Shen, Eddie Q. Yan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy.arXiv preprint arXiv:1802.04799 11, no. 2018 (2018): 20.


2024 
[Enhancing Performance Through Control-Flow Unmerging and Loop Unrolling on GPU](https://ieeexplore.ieee.org/document/10444819)
A. Murtovi, G. Georgakoudis, K. Parasyris, C. Liao, I. Laguna and B. Steffen, cgo  2024


https://escholarship.org/uc/item/3rt0n0q2
Gal, A., Probst, C. W, & Franz, M. (2003). A denial of service attack on the Java bytecode verifier. UC Irvine: Donald Bren School of Information and Computer Sciences. Retrieved from https://escholarship.org/uc/item/3rt0n0q2


https://dl.acm.org/doi/pdf/10.1145/3620665.3640366
PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation ASPLOS '24: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2


/https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/448997590_1496256481254967_2304975057370160015_n.pdf?_nc_cat=106&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=4Yn8V9DFdbsQ7kNvgEwOdGk&_nc_ht=scontent-bos5-1.xx&oh=00_AYD-0YTCXuS11WU8rqC3N2aA-AfiflOptch_BD__V1V3xA&oe=6684630D
Meta Large Language Model Compiler: Foundation Models of Compiler Optimization Chris Cummins†, Volker Seeker†, Dejan Grubisic, Baptiste Rozière, Jonas Gehring, Gabriel Synnaeve, Hugh Leather†



Chlorophyll: Synthesis-Aided Compiler for Low-Power Spatial Architectures
Phitchaya Mangpo Phothilimthana, Tikhon Jelvis, Rohin Shah, Nishant Totla, Sarah Chasins, and Rastislav Bodik. PLDI 2014.



MLIR: A Compiler Infrastructure for the End of Moore’s Law
Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. arXiv preprint, 2020.


Trace-Based Just-in-Time Type Specialization for Dynamic Languages
Andreas Gal, Brendan Eich, Mike Shaver, David Anderson, David Mandelin, Mohammad R. Haghighat, Blake Kaplan, Graydon Hoare, Boris Zbarsky, Jason Orendorff, Jesse Ruderman, Edwin W. Smith, Rick Reitmaier, Michael Bebenita, Mason Chang, and Michael Franz. PLDI 2009.


Mesh: Compacting Memory Management for C/C++ Applications
Bobby Powers, David Tench, Emery D. Berger, and Andrew McGregor. PLDI 2019.

A Unified Theory of Garbage Collection
David F. Bacon, Perry Cheng, and V. T. Rajan. OOPSLA 2004.

Type-Based Alias Analysis
Amer Diwan, Kathryn S. McKinley, and J. Eliot B. Moss.

Aho, Alfred V., Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman. "Compilers: Principles, Techniques, and Tools."
Alpern, Bowen, Mark N. Wegman, and F. Kenneth Zadeck. "Detecting equality of variables in programs." ACM SIGPLAN Notices 23.7 (1988): 1-11.
Bodik, Rastislav, Rajiv Gupta, and Vivek Sarkar. "ABC: Path-sensitive dynamic test generation." ACM SIGPLAN Notices 35.5 (2000): 61-73.
Chaitin, Gregory J., et al. "Register allocation via coloring." Computer languages 6.1 (1981): 47-57.
Cooper, Keith D., and Linda Torczon. "Tiling for improved register usage." ACM SIGPLAN Notices 28.6 (1993): 279-290.
Cytron, Ron, et al. "Efficiently computing static single assignment form and the control dependence graph." ACM Transactions on Programming Languages and Systems (TOPLAS) 13.4 (1991): 451-490.
Ertl, M. Anton. "Threaded code." ACM Computing Surveys (CSUR) 32.2 (2000): 290-318.
Ferrante, Jeanne, Karl J. Ottenstein, and Joe D. Warren. "The program dependence graph and its use in optimization." ACM Transactions on Programming Languages and Systems (TOPLAS) 9.3 (1987): 319-349.
Ganapathi, Madhusudhan, et al. "Experience with the MIPS compiler." ACM SIGPLAN Notices 21.7 (1986): 175-187.
Goldberg, David. "What every computer scientist should know about floating-point arithmetic." ACM Computing Surveys (CSUR) 23.1 (1991): 5-48.
Hall, Mary W., and Ken Kennedy. "Efficient call graph analysis." ACM Letters on Programming Languages and Systems (LOPLAS) 1.3 (1992): 227-242.
Johnson, Mark. "Attribute grammars and the lambda calculus." ACM SIGPLAN Notices 18.6 (1983): 39-45.
Kennedy, Ken, and Kathryn S. McKinley. "Loop distribution with arbitrary control flow." ACM SIGPLAN Notices 29.6 (1994): 140-151.
Knoop, Jens, Oliver Rüthing, and Bernhard Steffen. "Lazy code motion." ACM SIGPLAN Notices 27.7 (1992): 224-234.
Lamport, Leslie. "The parallel execution of DO loops." Communications of the ACM 17.2 (1974): 83-93.
Muchnick, Steven S. "Advanced compiler design and implementation." Elsevier, 1997.

Sarkar, Vivek. "Partitioning parallel programs for macro-dataflow." ACM SIGPLAN Notices 23.7 (1988): 98-106.
Shivers, Olin. "Control-flow analysis in Scheme." ACM SIGPLAN Notices 23.7 (1988): 164-174.
Steensgaard, Bjarne. "Points-to analysis in almost linear time." ACM SIGPLAN Notices 31.5 (1996): 32-41.
Tarjan, Robert E. "Depth-first search and linear graph algorithms." SIAM journal on computing 1.2 (1972): 146-160.
Tichy, Walter F. "Smart recompilation." ACM Transactions on Programming Languages and Systems (TOPLAS) 8.3 (1986): 273-291.
Wolf, Michael E., and Monica S. Lam. "A data locality optimizing algorithm." ACM SIGPLAN Notices 26.6 (1991): 30-44.
Yaccarino, Joseph, and Keshav Pingali. "Data-flow analysis for distributed-memory multiprocessors." ACM SIGPLAN Notices 27.9 (1992): 353-363.
Zadeck, F. Kenneth, and Olivier Rüthing. "Incremental data flow analysis." ACM SIGPLAN Notices 23.7 (1988): 132-146.


Here is a revised list of 25 important papers, focusing on middle-end optimizations, broader compiler topics, garbage collection, and parallelism, and excluding books:


2. Bodik, Rastislav, Rajiv Gupta, and Vivek Sarkar. "ABC: Path-sensitive dynamic test generation." ACM SIGPLAN Notices 35.5 (2000): 61-73.
3. Chaitin, Gregory J., et al. "Register allocation via coloring." Computer languages 6.1 (1981): 47-57.
4. Cooper, Keith D., and Linda Torczon. "Tiling for improved register usage." ACM SIGPLAN Notices 28.6 (1993): 279-290.
5. Cytron, Ron, et al. "Efficiently computing static single assignment form and the control dependence graph." ACM Transactions on Programming Languages and Systems (TOPLAS) 13.4 (1991): 451-490.
6. Ferrante, Jeanne, Karl J. Ottenstein, and Joe D. Warren. "The program dependence graph and its use in optimization." ACM Transactions on Programming Languages and Systems (TOPLAS) 9.3 (1987): 319-349.
7. Ganapathi, Madhusudhan, et al. "Experience with the MIPS compiler." ACM SIGPLAN Notices 21.7 (1986): 175-187.
8. Goldberg, David. "What every computer scientist should know about floating-point arithmetic." ACM Computing Surveys (CSUR) 23.1 (1991): 5-48.
9. Hall, Mary W., and Ken Kennedy. "Efficient call graph analysis." ACM Letters on Programming Languages and Systems (LOPLAS) 1.3 (1992): 227-242.
10. Kennedy, Ken, and Kathryn S. McKinley. "Loop distribution with arbitrary control flow." ACM SIGPLAN Notices 29.6 (1994): 140-151.
11. Knoop, Jens, Oliver Rüthing, and Bernhard Steffen. "Lazy code motion." ACM SIGPLAN Notices 27.7 (1992): 224-234.
12. Lamport, Leslie. "The parallel execution of DO loops." Communications of the ACM 17.2 (1974): 83-93.
13. McKinley, Kathryn S., Steve Carr, and Chau-Wen Tseng. "Improving data locality with loop transformations." ACM Transactions on Programming Languages and Systems (TOPLAS) 18.4 (1996): 424-453.

15. Sarkar, Vivek. "Partitioning parallel programs for macro-dataflow." ACM SIGPLAN Notices 23.7 (1988): 98-106.
16. Shivers, Olin. "Control-flow analysis in Scheme." ACM SIGPLAN Notices 23.7 (1988): 164-174.
17. Steensgaard, Bjarne. "Points-to analysis in almost linear time." ACM SIGPLAN Notices 31.5 (1996): 32-41.
18. Tarjan, Robert E. "Depth-first search and linear graph algorithms." SIAM journal on computing 1.2 (1972): 146-160.
19. Tichy, Walter F. "Smart recompilation." ACM Transactions on Programming Languages and Systems (TOPLAS) 8.3 (1986): 273-291.
20. Wolf, Michael E., and Monica S. Lam. "A data locality optimizing algorithm." ACM SIGPLAN Notices 26.6 (1991): 30-44.
21. Appel, Andrew W. "Simple generational garbage collection and fast allocation." Software: Practice and Experience 19.2 (1989): 171-183.
22. Boehm, Hans-Juergen, and Mark Weiser. "Garbage collection in an uncooperative environment." Software: Practice and Experience 18.9 (1988): 807-820.
23. Dijkstra, Edsger W., et al. "On-the-fly garbage collection: An exercise in cooperation." Communications of the ACM 21.11 (1978): 965-975.
24. Bacon, David F., Perry Cheng, and V. T. Rajan. "A real-time garbage collector with low overhead and consistent utilization." ACM SIGPLAN Notices. Vol. 38. No. 5. 2003.
25. Lea, Doug. "A Java fork/join framework." Proceedings of the ACM 2000 conference on Java Grande. 2000.


Here is a list of 15 papers that discuss the differences and challenges between compiling for CPUs and GPUs, focusing on performance optimization, memory management, parallelism, and execution models:

1. **Fatahalian, Kayvon, and Mike Houston.** "A closer look at GPUs." *Communications of the ACM* 51.10 (2008): 50-57.
2. **Garland, Michael, et al.** "Parallel computing experiences with CUDA." *IEEE Micro* 28.4 (2008): 13-27.
3. **Ryoo, Seongbeom, et al.** "Optimization principles and application performance evaluation of a multithreaded GPU using CUDA." *Proceedings of the 13th ACM SIGPLAN Symposium on Principles and practice of parallel programming.* 2008.
4. **Hong, Sunpyo, and Hyesoon Kim.** "An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness." *ACM SIGARCH Computer Architecture News.* Vol. 37. No. 3. 2009.
5. **Baskaran, Muthu Manikandan, et al.** "Automatic data movement and computation mapping for multi-level parallel architectures with explicitly managed memories." *Proceedings of the 13th ACM SIGPLAN Symposium on Principles and practice of parallel programming.* 2010.
6. **Diamos, Gregory, and Sudhakar Yalamanchili.** "Harmony: an execution model and runtime for heterogeneous many core systems." *Proceedings of the 17th international symposium on High performance distributed computing.* 2008.
7. **Kerr, Andrew, Gregory Diamos, and Sudhakar Yalamanchili.** "A characterization and analysis of PTX kernels." *IEEE International Symposium on Workload Characterization (IISWC).* 2009.
8. **Wong, Hing-Cheong, et al.** "Pangaea: a tightly-coupled IA32 heterogeneous chip multiprocessor." *ACM SIGARCH Computer Architecture News.* Vol. 36. No. 3. 2008.
9. **Magni, Alessandro, et al.** "Exploiting user-defined kernel threads on GPUs." *ACM Transactions on Architecture and Code Optimization (TACO)* 8.3 (2011): 1-29.
10. **Pai, Siddhartha, et al.** "Improving GPGPU concurrency with elastic kernels." *Proceedings of the eighteenth international conference on Architectural support for programming languages and operating systems.* 2013.
11. **Zhang, Yixin, et al.** "Synk: a framework for optimal synchronization selection." *Proceedings of the 19th International Conference on Architectural Support for Programming Languages and Operating Systems.* 2014.
12. **Baghsorkhi, S. Shams, et al.** "An adaptive performance modeling tool for GPU architectures." *ACM SIGPLAN Notices* 45.5 (2010): 105-114.
13. **Boyd, Eric, et al.** "Characterization of performance of data-parallel kernels on Intel's experimental single-chip cloud computer." *Proceedings of the 2011 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS).* 2011.
14. **Aila, Timo, and Samuli Laine.** "Understanding the efficiency of ray traversal on GPUs." *Proceedings of the conference on High Performance Graphics 2009.* 2009.
15. **Holewinski, Jan, Louis-Noël Pouchet, and P. Sadayappan.** "High-performance code generation for stencil computations on GPU architectures." *Proceedings of the 26th ACM international conference on Supercomputing.* 2012.

These papers provide insights into the challenges and differences in compiling for CPUs and GPUs, highlighting the architectural differences and how they influence compilation strategies and performance optimizations.


Baghdadi, S., Größlinger, A. and Cohen, A., 2010. Putting automatic polyhedral compilation for GPGPU to work. In Proceedings of the 15th Workshop on Compilers for Parallel Computers (CPC'10).

Pradelle, B., Baskaran, M., Henretty, T., Meister, B., Konstantinidis, A. and Lethin, R., 2016, September. Polyhedral compilation for energy efficiency. In 2016 IEEE High Performance Extreme Computing Conference (HPEC) (pp. 1-7). IEEE.

Merouani, M., Boudaoud, K.A., Aouadj, I.N., Tchoulak, N., Bernou, I.K., Benyamina, H., Tayeb, F.B.S., Benatchba, K., Leather, H. and Baghdadi, R., 2024. LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers. arXiv preprint arXiv:2403.11522.


	Wegman & Zadeck, Constant Propagation with Conditional Branches, ACM Transactions on Programming Languages and Systems, 13(2):181-210, April 1991.


    	P. Briggs, K. D. Cooper, L. Taylor Simpson, Value Numbering, Software-Practice & Experience, 27(6): 701-724, 1997


        	K. Hazelwood, and D. Grove, Adaptive Online Context-Sensitive Inlining Conference on Code Generation and Optimization, pp. 253-264, San Francisco, CA March 2003.

    	X. Huang, S. M. Blackburn, K. S. McKinley, J. E. B. Moss, Z. Wang, and P. Cheng, The Garbage Collection Advantage: Improving Program Locality, ACM Conference on Object Oriented Programming, Systems, Languages, and Applications (OOPSLA), pp. 69-80, Vancouver, Canada, October 2004.


        	McKinley, Carr, & Tseng, Improving Data Locality with Loop Transformations, ACM Transactions on Programming Languages and Systems, 18(4):424-453, July 1996.


            https://cfallin.org/blog/2021/03/15/cranelift-isel-3/  cranelift checking correctness in register allocator 
            uses data flow, value numbering and fuzzing 

            Retargeting and Respecializing GPU Workloads for Performance Portability
            Ivanov, I.R., Zinenko, O., Domke, J., Endo, T. and Moses, W.S., 2024, March. Retargeting and Respecializing GPU Workloads for Performance Portability. In 2024 IEEE/ACM International Symposium on Code Generation and Optimization (CGO) (pp. 119-132). IEEE.