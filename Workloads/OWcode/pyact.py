import sys

def main(args):
    # Check if the 'number' argument is provided
    if 'number' in args:
        number = args['number']
        
        # Convert the input to an integer
        try:
            number = int(number)
        except ValueError:
            return {"error": "Invalid input. 'number' must be a numeric value."}
        
        # Perform different actions based on the value of 'number'
        if number == 1:
            result = "number 1 was provided"
        elif number == 2:
            result = "number 2 was provided"
        elif number == 3:
            result = "number 3 was provided"
        elif number == 4:
            result = "number 4 was provided"
        elif number == 5:
            result = "number 5 was provided"
        else:
            return {"error": "Invalid input. 'number' must be in the range (1, 2, 3, 4, 5)."}
        
        return {"result": result}
    else:
        return {"error": "Missing 'number' argument."}