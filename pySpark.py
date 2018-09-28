a = [1,2,3,4]

def add(num):
	return num + 1 
# print map(add,a)


def isEven(number):
	return number%2 == 0
# print filter(isEven,a)

#reduce: applies a function to all paris of elements of a list, 
# to returns only 1 element
def add(x,y):
	return x + y
# print reduce(add,a)

#Some exercises:
#get second elements of each tuple
a = [(1,2),(3,4),(5,6)]
# print map(lambda x: x[1], a)

#get sum of the seconds elements
# print reduce(lambda x,y: x + y, map(lambda x: x[1], a))

'''
How to start a Spark folder.
1. download spark
2. Go to that folder and do $sudo /bin/pyspark
3. sc tells spark where to connect
'''

#creat an RDD
sc.parallelize(range(1,10)) #go from list to RDD
#get first in the RDD
sc.parallelize(range(1,10)).first()
#convert values
sc.parallelize(range(1,10)).map(lambda x: x + 1).first()
sc.parallelize(range(1,10)).map(lambda x: x + 1).collect()
sc.parallelize(range(1,10)).map(lambda x: x + 1).reduce(lambda x,y: x + y)
sc.parallelize(range(1,10)).map(lambda x: x + 1).filter(lambda x: x%2 ==0).reduce(lambda x,y: x + y)



'''
operate with file
'''
people = sc.textFile("somebook.txt")

