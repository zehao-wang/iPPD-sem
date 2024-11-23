import timeit
def execution_time(method):
    """ decorator style """

    def time_measure(*args, **kwargs):
        ts = timeit.default_timer()
        result = method(*args, **kwargs)
        te = timeit.default_timer()

        print(f'Excution time of method {method.__qualname__} is {te - ts} seconds.')
        #print(f'Excution time of method {method.__name__} is {te - ts} seconds.')
        return result

    return time_measure