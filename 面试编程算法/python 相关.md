# python 相关
## Tips
1. 在异常处理的 except block 中，把异常赋予了一个一个变量，那么这个变量会在 except block 执行结束时被删除。
2. 当全局变量指向的对象不可变时，比如是整型、字符串等等，如果你尝试在函数内部改变它的值，却不加关键字 global，就会抛出异常；如果全局变量指向的对象是可变的，比如是列表、字典等等你就可以在函数内部修改它了。
3. '=='操作符比较对象之间的值是否相等，而'is'操作符比较的是对象的身份标识是否相等，即它们是否是同一个对象，是否指向同一个内存地址。在 Python 中，每个对象的身份标识，都能通过函数 id(object) 获得。因此，'is'操作符，相当于比较对象之间的 ID 是否相等。**比较操作符'is'的速度效率，通常要优于'=='**。
4. 浅拷贝（shallow copy）与深拷贝（deep copy）：
    - **浅拷贝**是指重新分配一块内存，创建一个新的对象，里面的元素是原对象中子对象的引用。因此，如果子对象是可变的，改变后也会影响拷贝得到的新对象，存在**副作用**。常见的浅拷贝方法是使用数据类型本身的构造器；对于可变的序列，我们还可以通过切片操作符':'完成浅拷贝；还可以使用copy.copy()。
      ```
      # 浅拷贝
      import copy
      l1 = [1,2,3]
      l2 = list(l1)
      l3 = l1[:]
      l4 = copy.copy(l1)
      ```
    - *但是*，对于元组，使用上述三种方法不会创建浅拷贝，会返回一个指向相同元组的**引用**。
    - **深拷贝**是指重新分配一块内存，创建一个新的对象，并且将原对象中的子对象以递归的方式在新对象中创建新的子对象。新对象和原对象没有任何关联：
      ```
      # 深拷贝
      import copy
      l1 = [[1,2],(3,4),5]
      l2 = copy.deepcopy(l1)
      
      ```
    - python的深拷贝中会维护一个字典，记录已经拷贝的对象及其ID，提高效率并防止无限递归的发生。
5. C++里：**值传递**：拷贝参数的值，传递给函数里的新变量，二者相互独立。**引用传递**：把参数的引用传递给新变量，二者指向同一内存地址。而python里参数传递是**赋值传递（pass by assignment）**，参数传递时，只是让新变量与原变量指向相同的对象。
6. python的数据类型，如int，string等是不可变的，$a = a + 1$并不是让$a$的值增加1，而是创建了一个值为2的对象，并让$a$指向它。
7. python的赋值“=”并不是创建了新对象，而是同一个对象被多个变量指向或引用。可变对象（list, set, dict等）的改变会影响所有指向该对象的变量。更新不可变对象（int，string，tuple等）时，会返回一个新的对象。
8. 在 Python 中，使用了 yield 的函数被称为**生成器（generator）**。生成器（generator）并不会像**迭代器（iterator）**一样占用大量内存，只有在被使用时才会调用。迭代器是一个有限集合，而生成器可以成为无限集。
9. 把需要排序的属性拿出来作为一个 tuple，主要的放前面，次要的放后面。
    ```
    # 例如
    lst = [1, -2, 10, -12, -4, -5, 9, 2]
    lst.sort(key=lambda x: (x < 0, abs(x)))
    print(lst)
    >>>[1, 2, 9, 10, -2, -4, -5, -12]
    ```
10. **%time** 是 jupyter notebook 自带的语法糖，用来统计一行命令的运行时间

## 并发编程
### 协程
1. 协程是实现并发编程的一种方式，协程为单线程，由用户决定在哪些地方交出控制权切换到下一任务。
2. asyncio库中的asyncio.create_task，asyncio.run 这些函数都是 Python 3.7 以上的版本才提供的。
### **并发（concurrency） vs. 并行（parallelism）**
1. 在 Python 中，并发并不是指同一时刻有多个操作（thread、task）同时进行。相反，某个特定的时刻，它只允许有一个操作发生，只不过线程 / 任务之间会互相切换，直到完成。
2. thread 和 task 两种切换顺序的不同方式，分别对应Python 中并发的两种形式——threading 和 asyncio。
3. 并行指的是同时进行。python中的multi-processing即并行操作：比如你的电脑是 6 核处理器，那么在运行程序时，就可以强制Python 开 6 个进程，同时执行，以加快运行速度。
4. 并发通常用于I/O频繁操作的场景；并行通常用于CPU heavy的场景。
### 单线程与多线程
1. 单线程的优点是简单明了，但效率低下
## Decorator
装饰器就是通过装饰器函数来修改原函数的一些功能，避免对原函数的直接修改。提高程序可重用性，降低耦合度，提高开发效率。
1. 函数是一等公民（first-class citizen），可以赋给变量，可以当做参数传给另一个函数，可以在函数里嵌套函数，函数的返回值可以是函数对象（闭包）。
    ```
    # 函数嵌套
    def func(message):
      def print_message(massage):
        print('it is {}'.format(message))
      return print_message(message)

    func('a secret')

    >>> it is a secret

    # 闭包
    def func_closure():
      def print_message(message):
        print('it is '+message)
      return print_message

    a = func_closure()
    a('a secret')

    >>> it is a secret
    ```
2. @ —— 语法糖(syntactic sugar)，在需要装饰的函数上方加上@decorator即可。
3. 通常会把$*args$和$**kwargs$作为装饰器内部函数wrapper()的参数，表示接受任意类型和数量的参数。
4. 内置的装饰器 @functools.wrap 会帮助保留原函数的元信息。
5. 类也可以作为装饰器。类装饰器主要依赖于函数__call__()。
6. 装饰器的嵌套：
    ```
    @decorator1
    @decorator2
    @decorator3
    def func():
    # 等价于：
    decorator1(decorator2(decorator3(func)))
    ```
7. 实例：日志记录——计算某个函数的运行时间
    ```
    imort time
    import functools

    def log_exe_time():
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('{} takes {}ms.'.format(func.__name__, (end-start)*1000))
        return res
      return wrapper

    @log_exe_time
    def function():
      pass
    ```
## OOP
在python中，一切皆对象，对象的抽象就是类，对象的集合就是容器。list、tuple、set、dict都是容器。所有的容器都是可迭代的。
1. __init__表示构造函数，意即一个对象生成时会被自动调用的函数。
2. 如果一个属性以__开头，我们就默认这个属性是私有属性。私有属性，是指不希望在类之外的的地方被访问和修改的属性。
3. 用全大写来表示常量,和函数并列地声明并赋值
4. 类函数、成员函数和静态函数。
    - 静态函数：不涉及对象的私有变量（没有self作为参数），相同的输入能够得到完全性相同的输出。可以通过在函数前一行加上 @staticmethod 来表示。
    - 类函数的第一个参数一般为 cls，表示必须传一个类进来。类函数需要装饰器 @classmethod 来声明。
    - 成员函数则是我们最正常的类的函数，它不需要任何装饰器声明，第一个参数 self 代表当前对象的引用，可以通过此函数，来实现想要的查询 / 修改类的属性等功能。
5. 每个类都有构造函数，继承类是不会自动调用父类的构造函数的，因此必须在init()函数里显式调用父类的构造函数。执行顺序是：子类构造函数-->父类构造函数
6. 继承的优势：减少重复的代码，降低系统的熵值（即复杂度）。
7. 抽象类是一种自上而下的设计风格，定义接口。
8. 多重继承中为了避免基类的构造函数被调用多次，应该使用super来调用父类的构造函数。
9. Python并没有真正的私有化支持，但可用下划线得到伪私有：
    - _xxx "单下划线 " 开始的成员变量叫做保护变量，意思是只有类对象和子类对象自己能访问到这些变量，需通过类提供的接口进行访问；
    - __xxx 类中的私有变量/方法名，只有类对象自己能访问，连子类对象也不能访问到这个数据。
    - __xxx__ 魔法函数，前后均有一个“双下划线” 代表python里特殊方法专用的标识，如 __init__() 代表类的构造函数。
10. 如果想要强行访问父类的私有类型，做法是 self._ParentClass__var，这是非常不推荐的 hacky method。
```
class A:
    __a = 1

class B(A):
    pass

b = B()
print(b._A__a)
```
## 模块化
1. 通过绝对路径与相对路径，可以import模块
2. Python 是脚本语言，和 C++、Java 最大的不同在于，不需要显式提供 main() 函数入口。
3. import 在导入文件的时候，会自动把所有暴露在外面的代码全都执行一遍。因此，如果你要把一个东西封装成模块，又想让它可以执行的话，必须将要执行的代码放在 if __name__ == '__main__'下面。
4. 在大型工程中模块化非常重要，模块的索引要通过绝对路径来做，而绝对路径从程序的根目录开始。
5. from module_name import * 会把 module 中所有的函数和类全拿过来，如果和其他函数名类名有冲突就会出问题；import model_name 也会导入所有函数和类，但是调用的时候必须使用 model_name.func 的方法来调用，等于增加了一层 layer，有效避免冲突。
