# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function

def static_vars(**kwargs):
    """
    Descriptions:
        : Set static variables in function.
        : if you want to get static vars, use func.var
    Args:
        **kwargs (Any=Any, ...): variables to use static
    """
    def decorate (func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


if __name__ == '__main__':
    @static_vars(counter=0)
    def foo():
        foo.counter += 1
        print("Counter is %d" % foo.counter)

    for _ in range(5):
        foo()

    print(foo.counter)
    print(foo.counter)