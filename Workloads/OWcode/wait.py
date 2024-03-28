from time import sleep


def main(args):
    if 'time' in args:
        time = args['time']

        try:
            time = int(time)
        except ValueError:
            return {"error": "Invalid input. 'time' must be a numeric value."}
        
        #sleep for the specified time
        sleep(time)
        return {"result": "waited for {} seconds".format(time)}