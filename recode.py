a = [1,2,3,4]

def add(num):
	return num + 1 
print map(add,a)


def isEven(number):
	return number%2 == 0
print filter(isEven,a)

#reduce: applies a function to all paris of elements of a list, 
# to returns only 1 element
def add(x,y):
	return x + y
print reduce(add,a)