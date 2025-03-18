def show_func_name(func):
    def wrapper(*args, **kwargs):
        print(f'\n\n\nRunning {func.__name__}...')
        return func(*args, **kwargs)
    return wrapper