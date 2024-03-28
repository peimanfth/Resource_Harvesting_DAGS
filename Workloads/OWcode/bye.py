def main(args):
    name = args.get("name", "stranger")
    bye = "Bye " + name + "!"
    print(bye)
    return {"GoodBye": bye}