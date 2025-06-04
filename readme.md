
## 使用说明

### 实验一 SIMD 

使用 `./sub.sh simd -O2` 提交运行 simd 的实验,开启-O2优化。 normal 、avx 、test 运行其他的。 其他优化选项也可以。

后缀 simd 是使用 neon 指令集的；

不同版本都新开了 main.cpp  md5.h  md5.cpp 这三个文件， 



# 碎碎念

把服务器上已经有的删掉了。
此外，那个input 下面的似乎也没用？ 在服务器里有？

发到群里的 框架里， main.cpp 与服务器上有一点不同， 此外多了个test.exe 不知道做什么的； 
然后input 文件 在服务器上没有，似乎放在别的地方了？

把框架复制了一份，比对后， 除了上述三个地方之外，没有区别。
因为进行了diff比对， 所以服务器上的应该就是原始版本， 没有被我不小心动过。
于是乎开始写吧。

~~似乎要记得每次写完之后 git clone 一份到本地？ 如果想保留历史提交记录的话？ 应该连着.git 文件夹一起打包就行吧。~~
本地写，然后git push 推送到服务器。


test.e 是 cerr 出来的 标准错误输出
test.o 是 cout 出来的 标准输出

提交时使用 test.sh 1 1 ， 对于SIMD实验  
第一个 1 表示 实验1(simd;若2,3,4则是后面几次实验)， 
第二个1 表示申请几个核心(?)

本地修改代码，然后 push到这里，进行实验；大概就这样的流程

## 乱七八糟的配置问题

intellisense需要指定 交叉编译器，不然代码提示不正确
git仓库... 乱七八糟

文件太多了，也许应该每个实验放一个文件夹里。

## 实验2 SIMD
我们需要 开一些文件，重新写那9个函数 ， 以及需要修改 main.cpp ，最后实现并行化...

向量寄存器有128位的。

FFGGHHII 没有写成宏函数 应该没关系

SIMD 版本hash 51s  正常版本 9s； 可能 用宏函数会更快？？

不是宏函数的问题， 把串行版本 换成 inline函数，并没有变慢。 (见md5_part_no_macro .h .cpp)
 
总算弄完了,果然还是用宏定义方便，不需要开那么多文件;文件一多就显得很乱。

使用 `./sub.sh simd -O2` 提交运行 simd 的实验,开启-O2优化。 normal 、avx 、test 运行其他的。


## 实验3 pthread

test.sh目前有三个参数，
    第一个参数代表实验编号，pthread对应编号2，openmp对应编号3；
    第二个参数对应申请核心数，多线程实验目前没有核心数申请要求，为1就行。
    第三个参数对应申请线程数，本次实验需要用到

讲真， 有gpt可以快速上手Cmake，直接用起来而不需要学。Cmake比sh脚本方便查看。

修改了 main 和 guessing 文件，此外 config.h 是配置， ThreadPool.h 是https://github.com/progschj/ThreadPool 的C++线程池。
测试
g++ -O2 guessing_pthread_pool.cpp train.cpp correctness_guess_pool.cpp md5.cpp -o main
实验
g++ -O2 main_pool.cpp train.cpp correctness_guess_pool.cpp md5.cpp -o main

## 实验4 MPI

没有在服务器上跑.
本地是使用cmake构建的，见CmakeLists.txt
只试过8进程。

