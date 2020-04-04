from test import Parent


def function(callbacks):
    Parent.a = 5
    callbacks[0].print()

print(Parent.a)

# c1 = C(1)
# c2 = C(2)

Parent.a = 6

l = [C(1), C(2)]

p(l)
# for elem in l:
#     elem.print()

Parent.a = 3
# for elem in l:
#     elem.print()

p(l)

