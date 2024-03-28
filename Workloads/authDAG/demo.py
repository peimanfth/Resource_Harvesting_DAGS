import composer

def authenticate(args):
    return {'value': args['password'] == 'abc123'}

def success(args):
    return {'message': 'success'}

def failure(args):
    return {'message': 'failure'}

def main():
    return composer.when(
        composer.action('authenticate',  { 'action': authenticate }),
        composer.action('success', { 'action': success }),
        composer.action('failure', { 'action': failure }))