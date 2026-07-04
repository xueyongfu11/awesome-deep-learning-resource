[TOC]



## 学习资源

- [从零Makefile落地算法大项目，完整案例教程](https://zhuanlan.zhihu.com/p/396448133)
- [Effective Modern cpp啃书-B站视频](https://space.bilibili.com/218427631/lists/3726019?type=series)

## 什么时候需要自己编写拷贝构造函数？

当编译器生成的“逐成员复制”不符合对象的复制语义时，需要自己编写拷贝构造函数。最典型的情况是：类直接管理裸指针指向的动态内存、文件描述符、Socket 或操作系统句柄等资源。

对于 `int`、`std::string`、`std::vector` 等普通成员，默认拷贝通常已经正确，不需要手写：

```cpp
class Person {
public:
    int age;
    std::string name;
};

Person a{18, "Tom"};
Person b = a;  // 默认拷贝即可
```

如果类直接管理裸指针，默认拷贝只会复制指针地址，而不会复制指针指向的对象：

```cpp
class Person {
public:
    Person() : age(new int(18)) {}

    ~Person() {
        delete age;
    }

private:
    int* age;
};
```

执行 `Person b = a;` 后，`a.age` 和 `b.age` 会指向同一块内存。两个对象析构时会重复释放这块内存。若期望两个对象各自拥有独立资源，就需要进行深拷贝：

```cpp
Person(const Person& other)
    : age(new int(*other.age)) {}
```

如果一个类需要自定义析构函数、拷贝构造函数或拷贝赋值运算符，通常还需要一起检查其他复制和移动操作，即 Rule of Five：

```cpp
~Person();
Person(const Person&);
Person& operator=(const Person&);
Person(Person&&) noexcept;
Person& operator=(Person&&) noexcept;
```

如果资源不应该被复制，应显式禁止复制：

```cpp
Person(const Person&) = delete;
Person& operator=(const Person&) = delete;
```

现代 C++ 更推荐使用 `std::string`、`std::vector` 和智能指针等资源管理类型，使默认拷贝和析构行为满足需求，这就是 Rule of Zero。

一句话总结：默认逐成员复制正确时不要手写；需要深拷贝、自定义共享语义或禁止复制时，才显式定义或删除拷贝操作。

## C++11 中 const 和 constexpr 有什么区别？

`const` 强调“对象初始化后不可修改”，`constexpr` 强调“表达式可以在编译期求值”。

```cpp
int read();

const int x = read();      // 正确：运行时初始化，之后不能修改
constexpr int y = 10 * 2;  // 正确：编译期即可确定
// constexpr int z = read();  // 错误：初始化器不是常量表达式
```

二者的主要区别如下：

| 特性 | `const` | `constexpr` |
| --- | --- | --- |
| 核心含义 | 对象不可被修改 | 可用于常量表达式 |
| 初始化 | 可以使用运行时值 | 必须使用编译期常量表达式 |
| 修饰变量 | 只保证只读 | 变量隐含顶层 `const` |
| 修饰成员函数 | 表示不修改当前对象 | 表示函数具备编译期求值能力；C++11 中非静态 `constexpr` 成员函数还隐含 `const` |

部分 `const` 变量也可以成为常量表达式。例如，使用常量表达式初始化的整型或枚举类型 `const` 变量，可以用于数组长度等编译期场景：

```cpp
const int size = 10;
int values[size];
```

但 `const` 本身并不保证值能在编译期确定：

```cpp
const int size = read();  // 只读，但不是编译期常量
```

`const` 放在非静态成员函数末尾，表示该函数不能通过 `this` 修改对象的非 `mutable` 成员：

```cpp
class User {
public:
    int age() const {
        return age_;
    }

private:
    int age_;
};
```

`constexpr` 函数在实参是常量表达式且满足相关规则时可以在编译期求值，但它也能像普通函数一样在运行时调用：

```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int a = square(5);  // 编译期求值
int n = read();
int b = square(n);            // 运行时求值
```

C++11 对普通 `constexpr` 函数体限制较严格，通常只能包含一条 `return` 语句；这些限制在后续标准中有所放宽。

一句话总结：只需要表达“初始化后不能修改”时使用 `const`；需要表达“可以参与编译期计算”时使用 `constexpr`。
