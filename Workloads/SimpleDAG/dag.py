import composer
# import inspect

def hello(env):
    return { 'message': 'hello' }
def byebye(args):
    return { 'res': 'byebye ' + args['message'] }
def main():
    # dic = {'action': hello}
    # exc = inspect.getsource(dic['action'])
    return(
        composer.sequence(
            composer.action('hello', { 'action': hello }),
            composer.action('byebye', { 'action': byebye })
        )
    )
# if __name__ == '__main__':
#     main()
#     print("done")
