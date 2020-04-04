
class Parent:
    a = 0


class C(Parent):
    def __init__(self, arg):
        self.b = arg

    def print(self):
        print(self.a, self.b)

