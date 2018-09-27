

#==========implementation of data structure ==============

#stack
class Stack:
	def __init__(self):
		self.data = []
	def add(self,data):
		self.data.append(data)
	def remove(self):
		return self.data.pop()

#Queue
class Queue:
	def __init__(self):
		self.data = []
	def add(self,data):
		self.data.append(data)
	def remove(self):
		return self.data.pop(0)

#implement hash table using arrays


#LinkedList
class LinkNode:
	def __init__(self,data):
		self.data = data
		self.next = None

class LinkedList:
	def __init__(self):
		self.head = None

	def addNode(self,data):
		n = Node(data)
		n.next = self.head
		self.head = n

	def printList(self):
		#traversing the linkedList
		current = self.head
		while current != None:
			print current.data
			current = current.next

#pointers is used in linked data structure

#binary tree
class TreeNode:
	def __init__(self,data):
		self.data = data
		self.left = None
		self.right = None
	def addTreeNode(self,newData):
		if newData < self.data:
			if self.left is None:
				self.left = TreeNode(newData)
			else:
				self.left.addTreeNode(newData)
		if newData > self.data:
			if self.right is None:
				self.right = TreeNode(newData)
			else:
				self.right.addTreeNode(newData)
	def printTreePostOrder(self):
		print self.data
		if self.left:
			self.printTreePostOrder(self.left)
		if self.right:
			self.printTreePostOrder(self.right)


#print all possible combination of binary numbers
def printBinary(n):
	if n == 1:
		return ["0","1"]
	result = set()
	for item in printBinary(n-1):
		result.add(item + "0")
		result.add(item + "1")

	return result
#print printBinary(3)


#tries
class Tries:
	def __init__(self):
		self.root = {}
	def insert(self,words):
		current = self.root
		for i in words:
			current = current.setdefault(i, {})
		current.setdefault("_end")
		print self.root
	def search(self,words):
		current = self.root
		for i in words:
			if i not in current:
				return false
			current = current[i]
		if "_end" in current:
			return True
		return False

#graph, dfs, bfs
class Graph:
	def __init__(self,connections):
		self.graph = {}
		self.addConnections(connections)
	def addConnections(self,connections):
		for node1,node2 in connections:
			self.added(node1,node2)
	def added(self,node1,node2):
		self.graph[node1].add(node2)
		self.graph[node2].add(node1)

def DFS(graph,start):
	visited,stack = set(), [start]
	while stack:
		node = stack.pop()
		if node not in visited:
			visited.add(node)
			stack.extend(graph[node] - visited)
	return visited

def BFS(graph,start):
	visited, queue = set(), [start]
	while queue:
		node = queue.pop(0)
		if node not in visisted:
			visisted.add(node)
			queue.extend(graph[node] - visited)
	return visited

#grid data structure
def generateGrid(row,col):
	return [[0 for i in range(col)] for y in range(row)]
def printGrid(grid):
	for i in range(len(grid)):
		print grid[i]
grid = generateGrid(2,4)
#printGrid(grid)

'''Data Structure
Array: insert O(n), delete O(n), access / added O(1)
Grid: 2 arrays
LinkedList : same operation as vector, but insert saved time, no need for extra memory. but access will
	take more time compared to array
Set : could add repeated items but the set won't take it. Can't access data in set, but could do add, remove,contains?
	set operation: set1 == set2, set1 != set2, set1 union set2, set1 intersect set2, set1 difference set2.
	use for loop to loop over set, don't use index:
		1. for i in set(): #do something => ok to do this
		2. for i in range(len(set())): #set does not maintain order
	set is actually a binary search tree

Map: key -> pair value stored, loop: for key,val in map.items()
	map[key] = pair
	del map[key]
	for i in map.keys()
	for i in map.items()
	E.g: anagram => all word of an anagram has the same sorted form

Recursion: asked for more stack which is limited by the design of hardware and OS system. 
	Exercise: use recursive to check palindrome


'''

#recursive example
def isPalindrome(word):
	if len(word) < 2:
		return True
	if word[0] == word[-1]:
		return isPalindrome(word[1:-1])
	else: return False


#recursive binary search

#exhaustive search and back tracking.

#priority queue from array and linkedList, print jobs. 

#heap: added data and slowly push it up to the priority. 

#Dijkstra's algorithm


#quick sort, and its bigO

#merge sort for O(nlogn)


#recursive binary search 

#String permutation, use recursion for exhaustive search

#AVL: height balanced tree, fetch, insert, delete, join with O(logn) => run time

#the n-queens problem
#use 1 array to represent position of the queen

#dice row to add up to 10
def diceRoll(dice,remaining,roll):
	'''assume that remaining < roll*6'''
	result = []
	if roll == 0:
		if remaining == 0:
			result.append(dice)
		return result
	if remaining > 6 :
		possible_dice = range(1,7)
	else:
		possible_dice = range(1,remaining)

	for dice in possible_dice:
		chosen = diceRoll(dice,remaining - dice, roll - 1)
		if chosen:
			for i in chosen:
				result.append([dice] + [i])
	return result

#print diceRoll(dice = None, remaining = 4, roll = 3)

# -------------------------------------------------------------------------
#dynamic programming:


#--------------Google Optimization Problems--------------------------------
#bin-packing
def knapSack(totalWeight, weight, value, n):
	#base case :
	if n== 0 or totalWeight == 0:
		return 0

	#if weight of the nth item is more than knapsack capacity W,
	#then this item cannot be included in the optimal solution
	if weight[n-1] > totalWeight:
		return knapSack(totalWeight,weight,value,n-1)
	else:
		# return the maximum of two cases:
    # (1) nth item included
    # (2) not included
		return max(val[n-1] + knapSack(totalWeight-weight[n-1], weight, val, n -1),
			knapSack(totalWeight,weight,val,n-1))

def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]
 
    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]
'''
weight = [10,20,30]
value = [2,5,7]
totalWeight = 30
n = len(value)
=> return total value 
'''
#---------------------#permutation of a string---------
def perms(s):        
	#s is a strinng
    if(len(s)==1): return [s]
    result=[]
    for i,v in enumerate(s):
        result += [v+p for p in perms(s[:i]+s[i+1:])]
    return result

#coins combinations 
cents = 25
denominations = [25, 10, 5, 1]
names = {25: "quarter(s)", 10: "dime(s)", 5 : "nickel(s)", 1 : "pennies"}

def count_combs(remaining, i, comb, add):
    if add: comb.append(add)
    if remaining == 0 or (i+1) == len(denominations):
        if (i+1) == len(denominations) and left > 0:
            comb.append( (remaining, denominations[i]) )
            i += 1
        while i < len(denominations):
            comb.append( (0, denominations[i]) )
            i += 1
        print (" ".join("%d %s" % (n,names[c]) for (n,c) in comb))
        return 1
    cur = denominations[i]
    return sum(count_combs(left-x*cur, i+1, comb[:], (x,cur)) for x in range(0, int(left/cur)+1))

print (count_combs(cents, 0, [], None))


#permutation of [1,2,3]

#all possible diceSum

#Djak search

#A* search

#Priority queue with heap implementation
class BinaryHeap:
  def __init__(self):
    self.items = [0]
  def __len__(self):
    return len(self.items) - 1
  
  #insert a new item into a tree
  def insert(self,k):
    self.items.append(k)
    self.percolate_up()
  def percolate_up(self):
    i = len(self.items)
    while i//2 > 0:
      if self.items[i] < self.items[i//2]:
        self.items[i // 2], self.items[i] = self.items[i], self.items[i // 2]
      i = i // 2

  #remove an item in a tree
  def delete(self):
    return_value = self.items[1]
    self.items[1] = self.items[len(self)]
    self.items.pop()
    self.percolate_down(1)
    return return_value

  def percolate_down(self,i):
    while i*2 <= len(self):
      min_child = self.min_child(i)
      if self.items[i] > self.items[min_child]:
        self.items[i], self.items[min_child] = self.items[min_child], self.items[i]
        i = min_child

  def min_child(self,i):
    if i*2 + 1 > len(self):
      return i*2
    if self.items[i*2] < self.items[i*2 +1]:
      return i*2
    return i*2 + 1
  
  #build min priority heap data structure
  def build_heap(self,someList):
    i = len(someList) // 2
    self.items = [0] + someList
    while i>0:
      self.percolate_down(i)
      i = i -1
  
  #print the heap data structure
  def printHeap(self):
    print self.items
#---------------------------------------------------------
#Sort a LinkedList
class LinkNode:
  def __init__(self,data):
    self.data = data
    self.next = None
class LinkedList:
  def __init__(self):
    self.head = None
  def added(self,data):
    node = LinkNode(data)
    node.next = self.head
    self.head = node
def printLinkedList(Link):
  current = Link.head
  while current != None:
    print current.data
    current = current.next

# write a function that sort a linkedList with merge sort
def mergeSort(head):
  if head is None or head.next is None:
    return head
  l1,l2 = divideLists(head)
  l1 = mergeSort(l1)
  l2 = mergeSort(l2)
  head = mergeLists(l1,l2)
  return head

def divideLists(head):
  slow = head
  fast = head
  if fast:
    fast = fast.next
  while fast:
    fast = fast.next
    if fast:
      fast = fast.next
      slow = slow.next
  mid = slow.next
  slow.next = None
  return head,mid
  
def mergeLists(link1, link2):
  temp = None
  if link1 is None:
    return link2
  if link2 is None:
    return link1
  if link1.data <= link2.data:
    temp = link1
    temp.next = mergeLists(link1.next, link2)
  else:
    temp = link2
    temp.next = mergeLists(link1, link2.next)
  return temp



#reverse a LinkedList
def reverseLinkedList(linkedList):
	current = linkedList.head
	previous = None
	next_node = None
	while current.next is not None:
		previous = current
		current = current.next
		next_node = current.next

		current.next = previous
		current = next_node

	return linkedList

#quick Sort
def quickSort(someList,start,stop):
	if stop - start < 1:
		return someList
	else:
		pivot = someList[start]

		left = start + 1
		right = stop
		while left  <= right :
			while someList[left] < pivot:
				left += 1
			while someList[right] > pivot:
				right -= 1
			if left < right:
				someList[left],someList[right] = someList[right],someList[left]
				left += 1
				right -=1
		someList[start],someList[right] = someList[right],someList[start]
		quickSort(someList,start,right)
		quickSort(someList,left,stop)

nlist = [14,46,43,41,45,21,70]
quickSort(nlist,0, len(nlist) - 1)
print nlist 

#merge Sort
def mergeSort(nlist):
    print("Splitting ",nlist)
    if len(nlist)>1:
        mid = len(nlist)//2
        lefthalf = nlist[:mid]
        righthalf = nlist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=j=k=0       
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                nlist[k]=lefthalf[i]
                i=i+1
            else:
                nlist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            nlist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            nlist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",nlist)

nlist = [14,46,43,27,57,41,45,21,70]
#quickSort(nlist,0, len(nlist) - 1)
#print(nlist)


#generate Grid

#AVL problem
class Graph:
  def __init__(self,connections):
    self.graph = {}
    self.addConnections(connections)
  def addConnections(self,connections):
    for node1,node2 in connections:
      self.added(node1,node2)
  def added(self,node1,node2):
    self.graph[node1].add(node2)
    self.graph[node2].add(node1)

def BFS(graph,start):
  visited, queue = set(), [start]
  while queue:
    vertex = queue.pop(0)
    if vertex not in visited:
      queue.append(graph[vertex] - visited)
  return visited

def DFS(graph, start):
  visited, stack = set(), [start]
  while stack:
    vertex = stack.pop()
    if vertex not in visited:
      stack.append(graph[vertex] - visited)
  return visited

class Tree:
  def __init__(self,data):
    self.root = data
    self.left = None
    self.right = None
  def addNode(self,data):
    if data < self.root:
      if self.left is None:
        self.left = Tree(data)
      else:
        self.left.addNode(data)
    if data > self.root:
      if self.right is None:
        self.right = Tree(data)
      else:
        self.right.addNode(data)
  def printTree(self):
    print self.root
    if self.left:
      self.left.printTree()
    if self.right:
      self.right.printTree()


class Tries:
  def __init__(self):
    self.root = {}
  def insert(self,word):
    current = self.root
    for i in word:
      if i in current:
        current = current[i]
      else:
        current[i] = {}
        current = current[i]
    current["_end"]= {}
    return
  def search(self,word):
    current = self.root
    for i in word:
      if i not in current:
        return False
      current = current[i]
    if "_end" in current:
      return True
    return False  
  def printTries(self):
    print self.root

#-----------------------------------------------------------

#find longest subarray with continuous sum
def findSubArrayWithLargestSum(array):
	total = 0
	start =0
	result = 0
	for i in range(len(array)):
		if array[i] > 0 :
			start = i
			total += array[i]
			result = total
		if total > 0 and array[i] < 0:
			start = 0
			total = 0
	return result
# print findSubArrayWithLargestSum([1,2,3])
# print findSubArrayWithLargestSum([-1,1,-3,2,3,-4])

#find a missing number in the array, given an array with a missing
#number, find that number
#Example:
#input: [1,2,3,4,6]
#output: 5
def findArrayMissingNumber(array):
	for i in range(len(array)-1):
		if array[i] + 1 != array[i+1]:
			return array[i] + 1
	return False

#find a middle element in a linked list
def findMiddle(linkedList):
	slowPointer = head
	fastPointer = head
	while fastPointer is not None:
		slowPointer = slowPointer.next
		fastPointer = fastPointer.next.next
	return slowPointer

#merge 2 sorted linked list
def merge2SortedLinkedList(link1,link2):
	current1 = link1.head
	current2 = link2.head
	while current1 is not None and current2 is not None:
		previous = current1
		if current2.data > previous.data:
			previous.next = current2
			current2 = current2.next
			current2.next = current1.next
		else:
			#do something like this for the other case
			pass
	return link1

#add 2 numbers represented by linkedlist

#check if linkedlist is palindrome
#method 1 use a stack:

#is a number palindrome:
def isPalindromeStackMethod(number):
	originalNumber = number
	factors = 0
	total = 0
	while number != 0:
		lastDigit = number % 10
		total += lastDigit*(10**factors)
		factors += 1
		number = number / 10
	return total == originalNumber
#is a linkedList palinedrome ?
def isPalindromeLink(link):
	#reverse the first half
	#if fast reach the end
	#traverse backward
	fast = link.head
	slow = link.head
	previous_slow = None
	placeHolder = None
	while fast is not None:
		fast = fast.next.next
		previous = slow
		slow = slow.next
		placeHolder = slow
		slow.next = previous
	while placeHolder is not None:
		if placeHolder.data != slow.data:
			return False
		placeHolder = placeHolder.next
		slow = slow.next
	return True

#swap linkedList pairwise:
def swapLinkedListPairWise(link):
	current = link.head
	previous = None
	temp = None
	while current is not None:
		previous = current
		current = current.next
		temp = current.next
		current.next = previous
		previous.next = temp
		current = temp
	return link

#sort an array of 0s , 1s, 2s: 3 way partitioning
def sort012(array):
	low = 0
	mid = 0
	end = len(array) - 1
	while mid <= end:
		if array[mid] == 0:
			array[low], array[mid] = array[mid],array[low]
			low += 1
			mid += 1
		elif array[mid] == 1:
			mid += 1
		elif array[mid] == 2:
			array[mid],array[end] = array[mid],array[end]
			end -=1
	return array

#equilibrium indexes is when sum of all lower indexes equal to sum of all
#higher indexes. 
def equilibriumIndex(array):
	leftSum = 0
	rightSum = 0
	leftIndex = 0
	rightIndex = -1
	while leftIndex < rightIndex:
		leftSum += array[leftIndex]
		rightSum += array[rightIndex]
		if leftSum == rightSum:
			return leftIndex + 1
	return -1


# #find maximum sub of subsequence in which subsequence must be in increasing
# def findMaxSumIncreasingSequence(array):
# 	total = 0
# 	indexSequence = [0 for x in range(len(array))]

# 	for i in range(len(array)):
# 		indexSequence[i] = array[i]

# 	#compute maximum sum values in bottom up manners
# 	for i


#leaders in an array: numbers that is greater than all of its right side
def printLeaders(array):
	for i in range(-1,-len(array)+1,-1):
		if array[i - 1] > array[i]:
			print "leaders, ", array[i - 1]
arr = [16, 17, 4, 3, 5, 2]
#printLeaders(arr)

#parenthesis checker:
def checkParenthesis(string):
	stack = []
	for i in string:
		if i in "({[":
			stack.append(i)
		else:
			if stack.pop()+ i in ["()", "{}", "[]"]:
				continue
			else:
				return False
	return True
#print checkParenthesis("({[]})")
#print checkParenthesis("({)}")

#check if a string is palindrome:
def checkPalindromeString(string):
	start = 0
	end = len(string) -1
	while start < end:
		if string[start] != string[end]:
			return False
		start += 1
		end -= 1
	return True

#recursively remove duplicates in a string

#longest common substrings

#longest common prefix

#longest distinct characters in the string

#form a palindrome
def palindromeCreator(string):
	total = ""+string
	for i in string:
		total = i + total
	return total
#print palindromeCreator("abc")


#-------------------Binary Tree algorithm------------------
class TreeNode:
	def __init__(self,data):
		self.root = data
		self.left = None
		self.right = None
#print left view of a binary tree
def printAllLeft(tree):
	if tree.left is None:
		return
	print tree.left.root
	printAllLeft(tree.left)
	printAllLeft(tree.right)

root = TreeNode(5)
root.left = TreeNode(1)
root.right = TreeNode(10)
root.right.left = TreeNode(3)
root.right.right = TreeNode(12)
#printAllLeft(root)


#check if the tree is binary
def checkBinaryTree(tree):
	if tree.left is None and tree.right is None:
		return True
	if tree.left.root > tree.root or tree.right.root < tree.root:
		return False
	return checkBinaryTree(tree.left) and checkBinaryTree(tree.right)
print checkBinaryTree(root)

#balanced tree is for  efficiency search and insert data, 
#less call stack needed
#AVL tree

#AVLNode
class AVLNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
		self.height = 1

class AVLTree:
	def __init__(self):
		self.root = None
	#recursive function to insert a new node
	def insert(self,root,key):
		#Step 1- perform normal BST insert
		if not root:
			return AVLNode(key)
		elif key < root.val:
			root.left = self.insert(root.left,key)
		else:
			root.right = self.insert(root.right,key)

		#step 2 - update the height of the ancestor node
		root.height = 1 + max(self.getHeight(root.left),self.getHeight(root.right))

		#step 3- get the balance factor
		balance = self.getBalance(root)

		#step 4- if the node is unbalanced, then try out the 4 cases
		#case 1 - Left left
		if balance > 1 and key < root.left.val:
			return self.rightRotate(root)

		#case 2 - Right RIght
		if balance < -1 and key > root.right.val:
			return self.leftRotate(root)

		#case 3 - left right
		if balance > 1 and key > root.left.val:
			root.left = self.leftRotate(root.left)
			return self.rightRotate(root)

		#case 4 - right left
		if balance < -1 and key < root.right.val:
			root.right = self.rightRotate(root.right)
			return self.leftRotate(root)
		return root

	def leftRotate(self, z):
 
        y = z.right
        T2 = y.left
 
        # Perform rotation
        y.left = z
        z.right = T2
 
        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                         self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                         self.getHeight(y.right))
 
        # Return the new root
        return y
 
    def rightRotate(self, z):
 
        y = z.left
        T3 = y.right
 
        # Perform rotation
        y.right = z
        z.left = T3
 
        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                        self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                        self.getHeight(y.right))
 
        # Return the new root
        return y

	def getHeight(self,root):
		if not root:
			return 0
		return root.height

	def getBalance(self,root):
		if not root:
			return 0
		return self.getHeight(root.left) - self.getHeight(root.right)

	#--------------delete implementation--------------
	#recursive function to delete a node with given key from subtree
	#with given root. It returns root of the modified subtree
	def delete(self,key,node = None):
		#step 1 - Perform normal BST
		if not root:
			return root
		elif key < root.val :
			root.left = self.delete(root.left,key)
		elif key > root.val :
			root.right = self.delete(root.right,key)
		else:
			if root.left is None:
				temp = root.right
				root = None
				return temp
			elif root.right is None:
				temp = root.left
				root = None
				return temp
			temp = self.getMinValueNode(root.right)
			root.val = temp.val
			root.right = self.delete(root.right,temp.val)

		#if the tree as only one node, simply return it
		if root is None:
			return root

		#step 2- update the height of the ancestor node
		root.height = 1 + max(self.getHeight(root.left), self.getHeight)

		#step 3 - get the balance factor
		balance = self.getBalance(root)

		#step 4 - if the node is unbalanced, then try out the 4 cases
		#case 1 - left left
		if balance > 1 and self.getBalance(root.left) >= 0:
			return self.rightRotate(root)
		#case 2 - right right
		if balance < -1 and self.getBalance(root.right) <= 0:
			return self.leftRotate(root)
		#case 3 - left right
		if balance > 1 and self.getBalance(root.left) <0 :
			root.left = self.leftRotate(root.left)
			return self.rightRotate(root)
		#case 4 - right left
		if balance < -1 and self.getBalance(root.right) >0 :
			root.right = self.rightRotate(root.right)
			return self.leftRotate(root)
		return root

#-------------------------------------

#count number of connected component in an undirected graph
#count number of islands in 2-D matrix
class Graph:
	def __init__(self,row,col,g):
		self.ROW = row
		self.COL = col
		self.graph = g

	#a function to check if a given cell (row,col)
	#can be included in DFS
	def isSafe(self,i,j, visited):
		#row number is in range. column number
		#is in range and value is 1
		#and not yet visited
		rowBoundaryCheck = i >= 0 and i <self.ROW
		columnBoundaryCheck = j >= 0 and j < self.COL

		return (rowBoundaryCheck and columnBoundaryCheck not visited[i][j] and self.graph[i][j])

	#a utility function to do DFS for a 2D boolean matrix. It only considers
	#the 8 neighbors as adjacent vertices
	def DFS(self,i,j,visited):
		#these arrays are used to get row and column number of 8 neighbors
		#of a given cell
		rowNeighbor = [-1, -1, -1,  0, 0,  1, 1, 1]
		colNeighbor = [-1,  0,  1, -1, 1, -1, 0, 1]

		#Mark this cell as visited
		visited[i][j] = True

		#recur for all connected neighbors:
		for k in range(8):
			if self.isSafe(i + rowNeighbor[k], j + colNeighbor[k],visited):
				self.DFS(i + rowNeighbor[k], j + colNeighbor[k],visited)

	#the main function that returns count of islands in a given boolean 2D matrix
	def countIslands(self):
		#make a boolean array to mark visited cells
		#initially all cells are unvisited
		visited= [[False for j in range(self.COL)] for i in range(self.ROW)]

		count = 0 
		for i in range(self.ROW):
			for j in range(self.COL):
				#if cell with value 1 is not visited yet, then new island found
				if visited[i][j] == False and self.graph[i][j] == 1:
					#visit all cells in this island and increment island count
					self.DFS(i,j,visited)
					count += 1
		return count


#find changes for a given number

#---------------diceRoll problem---------------------
resultRolls = []
dices = range(1,7)
def diceRoll(numRoll,remaining,picks):
	if numRoll == 0:
		if remaining == 0:
			return True
		return False
	for i in dices:
		picks.append(i)
		if diceRoll1(numRoll -1, remaining - i, picks):
			print picks
		picks.pop()

diceRoll(numRoll = 3, remaining = 5,picks = [])
#------------------------------------------------------


#detect cycle in a linkedList
#create 2 pointers, one moves 2x speed of the other, if they reach the same
#node => cycle, if they reach None before meet => not cycle
def detectCycle(linkList):
	fastPointer = linkList.head
	slowPointer = linkList.head
	while fastPointer is not None and slowPointer is not None:
		fastPointer = fastPointer.next.next
		slowPointer = slowPointer.next
		if slowPointer.data == fastPointer.data:
			return True
	return False


#delete a node in BST
def deleteNode(root,key):
	if root is None:
		return root

	#if the key to be deleted is smaller than the root, then it should
	#be on the left of the tree
	if key < root.key:
		root.left = deleteNode(root.left,key)
	if key > root.key:
		root.right = deleteNode(root.right,key)
	#if the key is the same as root
	else:
		#root with only left child
		if root.left is None:
			temp = root.right
			root = None
			return temp
		#root with only right child
		elif root.right is None:
			temp = root.left
			root = None
			return temp
		#root with 2 childs:
		#1. get inorder sucessor of the right subtree
		temp = minValueNode(root.right)
		#2. copy the successor into the current node
		root.key = temp.key
		#3. Delete the inorder successor
		root.right = deleteNode(root.right,temp.key)
	return root
def minValueNode(node):
	'''
	find the node with minimum value found in a given tree
	'''
	current = node
	#loop down to find the leftmost leaf
	while(current.left is not None):
		current = current.left
	return current

#-----------find distance to 2 nodes in bst---------
class Node:
	def __init__(self,data):
		self.data = data
		self.right = None
		self.left = None

def pathToNode(root,path,k):
	#base case handling
	if root is None:
		return False
	#append the node value in path
	path.append(root.data)
	#see if the k is same as root's data
	if root.data == k:
		return True
	#check if k is found in left or right sub-tree
	if (root.left != None and pathToNode(root.left,path,k)) or (root.right != None and pathToNode(root.right,path,k)):
		return True
	#if not present in subtree rooted with root, remove root from path and
	# return False
	path.pop()
	return False

def distance(root,data1,data2):
	if root:
		#store path corresponding to node1: data1
		path1 = []
		pathToNode(root,path1,data1)

		#store path corresponding to node2: data2
		path2 = []
		pathToNode(root,path2,data2)

		#iterate through the paths to find common path length
		i = 0
		while i < len(path1) and i < len(path2):
			#get out as soon as the path differs or any path's 
			# length get exhausted
			if path1[i] != path2[i]:
				break
			i += 1
		#get the path length by deductible the intersecting path length
		# or till LCA
		return (len(path1) + len(path2) -2*i)
	else:
		return 0
#_-----------------------------------------------------------


#Java Check List
'''1. Be able to architect a company HR system
2. 4 components of OOP: encapsulation(access modifiers), inheritance, abstraction, polymorphism.
3. Topics:
	i. Class
	ii. constructor
	iii. Object
	iv. access modifiers: public (can be used outside of the class), protected (could use the function but all classes must be in the same packages) , private (private attributes only access through setter and getter, only accessible within the class, private methods are meant to use within the class, for example, for internal purpose, no one needs to access this method outside of the class), final (keyword like constants, will never change after declaration), public static attribute(shared between objects), public static method is used to call the method without instaniation the object) for example:
Vehicle.getVehicle();
	v. method override (@Override) => change the method function in the inherited class
		method overloading => different methods of the same name called depends on the params
	vi. polymorphism
	vii. abstraction
	viii. interfaces
	ix. inheritance
x.abstract class

4. Create a class vehicle, its properties, constructor, create an object from this class, add setter and getter for the class if modified of the class needed
5. Car inherit from Vehicle, called parent class constructor super(param 1, param 2), call parent method super.method1()

6. Polymorphism: when car inherited from vehicle class
Vehicle a = new Vehicle();
Vehicle b = new Car();

7. abstraction: abstract class need an abstract function, abstract function is empty and must be overriden in children class.

8. interfaces: implements, multiple interfaces, all method must be implemented, make an example on interfaces. 
'''

'''DATABASE
1. DataBase Normalization: 3 steps
2. know how to build primary key and forgein key. 
'''

graph = {'s': {'a' :2, 'b': 1},
			'a' : {'s': 2,"b":4, "c":8},
			'b' : {'s': 4,'a': 2, 'd':2},
			'c' : {'a': 2, 'd': 7, 't': 4},
			'd' : {'b' : 1, 'c': 11, 't': 5},
			't': {}
			}

def dijkstra(graph,src,dest,visited = [], distances = {}, predecessors = {}):
	if src == dest:
		#we build the shortest path and display it
		path = []
		pred = dest
		while pred is not None:
			path.append(pred)
			pred = predecessors.get(pred,None)
	else:
		if src not in visited:
			distance[src] = 0
		#visit the neighbors
		for neighbor in graph[src]:
			if neighbor not in visited:
				newDistance = distances[src] + graph[src][neighbor]
				if newDistance < distange.get(neighbor,float('inf')):
					distances[neighbor] = newDistance
					predecessors[neighbor] = src
		#mark as visited 
		visited.append(src)

		#now that all neighbors have been visited: recurse select the non-visited node
		#with the lowest distance 'x' run Dijskstra with src = 'x'
		unvisited = {}
		for k in graph:
			if k not in visited:
				unvisited[k] = distances.get(k,float('inf'))
		x = min(unvisited, key = unvisited.get)
		dijkstra(graph,x,dest,visited,distances,predecessors)
#test cases:
dijkstra(graph,'s','t')
#=================================
#solve the N-queen problems:
N = 4
def printSolution(board):
	for row in range(N):
		for col in range(N):
			print board[row][col]
# A utility function to check if a queen can be placed on board[row][col]. Note that
#this function is called when "col" queens are already placed in coluns from
# 0 to col - 1. so we need to check only left side for the attacking queens

def isSafe(board,row,col):
	#check this row on left side
	for i in range(col):
		if board[row][i] == 1:
			return False

	#check upper diagonal on left side
	for i,j in zip(range(row, -1, -1), range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	#check lower diagonal on left side
	for i,j in zip(range(row,N,1), range(col,-1,-1)):
        if board[i][j] == 1:
            return False
 
    return True

def solveNQueenUtil(board,col):
	#if all queens are placed then return True
	if col >= N :
		return True

	#consider this column and try placing this queen in all rows one by one
	for i in range(N):
		if isSafe(board,i,col):
			#place this queen in board[i][col]
			board[i][col] = 1
			#recur to place rest of the queens
			if solveNQueenUtil(board,col + 1):
				return True

			#if placing queen in board[i][col] doesn't lead to a solution, then queen
			#from board[i][col]
			board[i][col] = 0

	#if the queen cannot be placed in any row in this col then return false
	return False

def solveNQ():
    board = [ [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]
             ]
 
    if solveNQUtil(board, 0) == False:
        print "Solution does not exist"
        return False
 
    printSolution(board)
    return True

#largest maximum sum of sub array:
def maxSubArraySum(a,size):
     
    max_so_far = 0
    max_ending_here = 0
     
    for i in range(0, size):
        max_ending_here = max_ending_here + a[i]
        if max_ending_here < 0:
            max_ending_here = 0
         
        # Do not compare for all elements. Compare only   
        # when  max_ending_here > 0
        elif (max_so_far < max_ending_here):
            max_so_far = max_ending_here
             
    return max_so_far




#========================Algos by categories=======================

'''
Found on this web page: https://www.geeksforgeeks.org/fundamentals-of-algorithms/
0. Graph algos:
	a. BFS and DFS: snake and ladder problem, topological sorting
	b. Minimum spanning Tree: disjoint set, Kruskal's minimum spanning TRee
	c. Shortest paths: Dijkstra, Floyd Warshall algo, A*
	d. Connectivity: count islands
	e. Hard problems: graph coloring, travelling salesman using minimum spanning tree
1. Branch and bound: 0/1 knapSack, N-queen, travelling salesman
2. search and sort: binary search, merge sort, quick sort, merge sort 
				linkedList, count 1's in a sorted binary array, dutch's flag problem
3. Greedy Algos: Dijkistra, minimum spanning tree, minimum numbers of coins, 
4. Dynamic Programming: coin change, maximum sum increasing subsequence, longest
			palindromic subsequence, 0/1 knapSack, largest sum contiguous sub array,
			dice sum, 
5. Pattern search: anagram sub string search
6. Back tracking: all permutation of a string, N-queens, rat in a maze
7. Math algos: fibbonacci, factorial, shuffle a given array, tower of Hanoi, replace all "0" with "5" in an input integer
'''

#=========================Algos by data structure==================
'''
1. Implement the following data structure and algos from scratch, including time/space analysis
Prim's minimum spanning tree
AVLTree: add, delete, inorder,preorder,postorder
Graph: add, delete, dfs, bfs
Tries data structure
Dijkstra on graph
BinaryHeap
Merge sort
Binary Search
Quick Sort
LinkedList
Stack/Queue
Dictionary using 2 separated arrays, arrays within arrays, linked list within array. 

2. LinkedList:
reverse a linkedlist
merge 2 sorted linked list
merge sort for linked list
reverse a linked list in groups of given size
detect cyclic linkedlist
union and intersection of 2 linkedList


3. Stack: 
reerse a stack using recursion
sort a stack using recursion

4. Queue:
implemenet priority Queue

5. Binary Tree:
Tree traversal: preorder, inorder,postorder
BFS and DFS in binary Tree
maximum depth of a tree
print nodes at k distance from root
correct the bst
find a pair with given sum in a balanced BST
merge 2 balanced binary search trees
binary tree to binary search tree conversion

6. Heap:
BinayHeap
fibonacci heap
heap sort

7. Hasing:
find pair in an array with given sum. 

8. Tries:
longest prefix matching

9. Array:
leaders in array
median of two sorted arrays
find missing number
majority element
merge 2 sorted array, different size
rotate an array

10. matrix:
Boolean matrix
'''

#disjoint set: data structure that support makeSet, union, findSet

#=====================arrays==============================
# 1. Delete LRU
from datetime import datetime


class LRUCacheItem(object):
    """Data structure of items stored in cache"""
    def __init__(self, key, item):
        self.key = key
        self.item = item
        self.timestamp = datetime.now()


class LRUCache(object):
    """A sample class that implements LRU algorithm"""

    def __init__(self, length, delta=None):
        self.length = length
        self.delta = delta
        self.hash = {}
        self.item_list = []

    def insertItem(self, item):
        """Insert new items to cache"""

        if item.key in self.hash:
            # Move the existing item to the head of item_list.
            item_index = self.item_list.index(item)
            self.item_list[:] = self.item_list[:item_index] + self.item_list[item_index+1:]
            self.item_list.insert(0, item)
        else:
            # Remove the last item if the length of cache exceeds the upper bound.
            if len(self.item_list) > self.length:
                self.removeItem(self.item_list[-1])

            # If this is a new item, just append it to
            # the front of item_list.
            self.hash[item.key] = item
            self.item_list.insert(0, item)

    def removeItem(self, item):
        """Remove those invalid items"""

        del self.hash[item.key]
        del self.item_list[self.item_list.index(item)]

    def validateItem(self):
        """Check if the items are still valid."""

        def _outdated_items():
            now = datetime.now()
            for item in self.item_list:
                time_delta = now - item.timestamp
                if time_delta.seconds > self.delta:
                    yield item
        map(lambda x: self.removeItem(x), _outdated_items())


#=====================linked list algos===================
#remove duplicate in a sorted linked list
def removeDupateInSortedLinkedList(link):
	current = link.head
	pointer = None
	while current is not None:
		if current.data == pointer.data:
			current = current.next
			continue
		if current.data == current.next.data:
			pointer = current
		current = current.next
			
#move last element to the front of linkedlist
def moveLastToFront(link):
	head = link.head
	current = link.head
	while current is not None:
		current = current.next
	head.next = current.next
	current.next = head
	return link

#segregate even node from odd node
def segregateEvenOdd(link):
	pass


#=======================BST====================

#lowest common ancestor in binary tree
# This function returns pointer to LCA of two given
# values n1 and n2
# This function assumes that n1 and n2 are present in
# Binary Tree
def findLCA(root, n1, n2):
     
    # Base Case
    if root is None:
        return None
 
    # If either n1 or n2 matches with root's key, report
    #  the presence by returning root (Note that if a key is
    #  ancestor of other, then the ancestor key becomes LCA
    if root.key == n1 or root.key == n2:
        return root 
 
    # Look for keys in left and right subtrees
    left_lca = findLCA(root.left, n1, n2) 
    right_lca = findLCA(root.right, n1, n2)
 
    # If both of the above calls return Non-NULL, then one key
    # is present in once subtree and other is present in other,
    # So this node is the LCA
    if left_lca and right_lca:
        return root 
 
    # Otherwise check if left subtree or right subtree is LCA
    return left_lca if left_lca is not None else right_lca


#pair sum in BST-------
#using extra space
stored = set()
def findSumPair(root,sum):
	remaining = sum - root.data
	if remaining in stored:
		return root.data , remaining
	stored.add(remaining)
	findSumPair(root.left)
	findSumPair(root.right)
#there another method that used 2 pointers, please find solution

#-----------find distance to 2 nodes in bst---------
class Node:
	def __init__(self,data):
		self.data = data
		self.right = None
		self.left = None

def pathToNode(root,path,k):
	#base case handling
	if root is None:
		return False
	#append the node value in path
	path.append(root.data)
	#see if the k is same as root's data
	if root.data == k:
		return True
	#check if k is found in left or right sub-tree
	if (root.left != None and pathToNode(root.left,path,k)) or (root.right != None and pathToNode(root.right,path,k)):
		return True
	#if not present in subtree rooted with root, remove root from path and
	# return False
	path.pop()
	return False

def distance(root,data1,data2):
	if root:
		#store path corresponding to node1: data1
		path1 = []
		pathToNode(root,path1,data1)

		#store path corresponding to node2: data2
		path2 = []
		pathToNode(root,path2,data2)

		#iterate through the paths to find common path length
		i = 0
		while i < len(path1) and i < len(path2):
			#get out as soon as the path differs or any path's 
			# length get exhausted
			if path1[i] != path2[i]:
				break
			i += 1
		#get the path length by deductible the intersecting path length
		# or till LCA
		return (len(path1) + len(path2) -2*i)
	else:
		return 0

#iterative solution factorial
def factorial(n):
	total = 1
	for i in range(1,n+1):
		total = total * i
	return total


#iterative tree traversal

class TreeNode:
	def __init__(self,data):
		self.data = data
		self.left = None
		self.right = None



temp = TreeNode(20)
temp.left = TreeNode(10)
temp.right = TreeNode(30)
temp.left.left = TreeNode(4)
temp.left.right = TreeNode(15)
def traversal(tree):
	print tree.data
	if tree.left:
		traversal(tree.left)
	if tree.right:
		traversal(tree.right)

def iterativeTraversal(root):
	stack = [root]
	while stack:
		node = stack.pop()
		print node.data
		if node.right:
			stack.append(node.right)
		if node.left:
			stack.append(node.left)

#======================
def anagram(word1,word2):
	return sort(word1) == sort(word2)

#sum of a binary tree
def sumOfTree(root):
	if root is None:
		return 0
	return root.data + sumOfTree(root.left) + sumOfTree(root.right)

#find the minimum depth of the tree
def minimumDepth(root):
	#base case
	if root.left is None and root.right is None:
		return 1
	if root.left:
		return 1 + minimumDepth(root.left)
	if root.right:
		return 1 + minimumDepth(root.right)
	return min(minimumDepth(root.left),minimumDepth(root.right)) + 1
	

#find max sum path in a tree
def findMaxUtil(root):
	#base case
	if root is None:
		return 0

	#left and right store maximum path sum going through left and right
	#child of root
	left = findMaxUtil(root.left)
	right = findMaxUtil(root.right)

	#max path for parent call of root. this Path must include at most 
	#one child of root
	max_single = max(max(left,right) + root.data, root.data)

	#max top represents the sum when the node under consideration is the 
	#root of the maxSum Path and no ancestor of root are there in max sum path
	max_top = max(max_single, left + right + root.data)

	#static variable to store the changes, store the maximum result
	findMaxUtil.res = max(findMaxUtil.res,max_top)
	return max_single

def findMaxSum(root):
	#initialize result
	findMaxUtil.res = 999999999
	#compute and return result
	findMaxUtil(root)
	return findMaxUtil.res

# Given a Binary Tree, we need to print the bottom view from left to right. 
# A node x is there in output if x is the bottommost node at its horizontal distance. 
# Horizontal distance of left child of a node x is equal to horizontal distance of x 
# minus 1, and that of right child is horizontal distance of x plus 1.
def findLeafNode(root):
	'''recursive style'''
	if root.left is None and root.right is None:
		print root.data
	if root.left:
		findLeafNode(root.left)
	if root.right:
		findLeafNode(root.right)
def findLeafNodeIterative(root):
	'''iterative style'''
	queue = [root]
	while queue:
		node = queue.pop(0)
		if node.left:
			queue.append(node.left)
		if node.right:
			queue.append(node.right)
		if node.left is None and node.right is None:
			print node.data


#check if a binary tree is a subtree of another tree
def areIdentical(root1,root2):
	if root1 is None and root2 is None:
		return True
	if root1 is None or root2 is None:
		return False
	return (root1.data == root2.data and areIdentical(root1.left,root2.left) and areIdentical(root1.right,root2.right))
def isSubTree(T,S):
	#base case
	if S is None:
		return True

	if T is None:
		return True

	#check the tree with root as a current node
	if areIdentical(T,S):
		return True

	#if the tree with root as current node doesn't match then try left
	#and right subtree one by one
	return isSubTree(T.left,S) or isSubTree(T.right,S)


#longest common subsequence
# A Naive recursive Python implementation of LCS problem
 
def lcs(X, Y, m, n):
 
    if m == 0 or n == 0:
       return 0;
    elif X[m-1] == Y[n-1]:
       return 1 + lcs(X, Y, m-1, n-1);
    else:
       return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n));
# Dynamic Programming implementation of LCS problem
 
# Dynamic Programming implementation of LCS problem
 
def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in xrange(m+1)]
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
#end of function lcs
 
 
# # Driver program to test the above function
# X = "AGGTAB"
# Y = "GXTXAYB"
# print "Length of LCS is ", lcs(X, Y)


#find a way out of a maze algo

# Python3 program to solve Rat in a Maze 
# problem using backracking 
 
# Maze size
N = 4
 
# A utility function to print solution matrix sol
def printSolution( sol ):
     
    for i in sol:
        for j in i:
            print(str(j) + " ", end="")
        print("")
 
# A utility function to check if x,y is valid
# index for N*N Maze
def isSafe( maze, x, y ):
     
    if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1:
        return True
     
    return False
 
""" This function solves the Maze problem using Backtracking. 
    It mainly uses solveMazeUtil() to solve the problem. It 
    returns false if no path is possible, otherwise return 
    true and prints the path in the form of 1s. Please note
    that there may be more than one solutions, this function
    prints one of the feasable solutions. """
def solveMaze( maze ):
     
    # Creating a 4 * 4 2-D list
    sol = [ [ 0 for j in range(4) ] for i in range(4) ]
     
    if solveMazeUtil(maze, 0, 0, sol) == False:
        print("Solution doesn't exist");
        return False
     
    printSolution(sol)
    return True
     
# A recursive utility function to solve Maze problem
def solveMazeUtil(maze, x, y, sol):
     
    #if (x,y is goal) return True
    if x == N - 1 and y == N - 1:
        sol[x][y] = 1
        return True
         
    # Check if maze[x][y] is valid
    if isSafe(maze, x, y) == True:
        # mark x, y as part of solution path
        sol[x][y] = 1
         
        # Move forward in x direction
        if solveMazeUtil(maze, x + 1, y, sol) == True:
            return True
             
        # If moving in x direction doesn't give solution 
        # then Move down in y dirxection
        if solveMazeUtil(maze, x, y + 1, sol) == True:
            return True
         
        # If none of the above movements work then 
        # BACKTRACK: unmark x,y as part of solution path
        sol[x][y] = 0
        return False
 
# Driver program to test above function
if __name__ == "__main__":
    # Initialising the maze
    maze = [ [1, 0, 0, 0],
             [1, 1, 0, 1],
             [0, 1, 0, 0],
             [1, 1, 1, 1] ]
              
    solveMaze(maze)


#longest increasing subsequent

def longestIncreasingSubsequent(arr):
    n = len(arr)
 
    # Declare the list (array) for LIS and initialize LIS
    # values for all indexes
    lis = [1]*n
 
    # Compute optimized LIS values in bottom up manner
    for i in range (1 , n):
        for j in range(0 , i):
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 :
                lis[i] = lis[j]+1
 
    return max(lis)
# end of lis function
 
# Driver program to test above function
arr = [10, 22, 9, 33, 21, 50, 41, 60]
print "Length of lis is", longestIncreasingSubsequent(arr)


#coin changes multiple solutions:
# Returns the count of ways we can sum
# S[0...m-1] coins to get sum n
def count(array, m, remaining ):
 
    # If n is 0 then there is 1
    # solution (do not include any coin)
    if (remaining == 0):
        return 1
 
    # If n is less than 0 then no
    # solution exists
    if (remaining < 0):
        return 0;
 
    # If there are no coins and n
    # is greater than 0, then no
    # solution exist
    if (m <=0 and remaining >= 1):
        return 0
 
    # count is sum of solutions (i) 
    # including S[m-1] (ii) excluding S[m-1]
    return count( array, m - 1, remaining ) + count( array, m, remaining - array[m-1] );

# Driver program to test above function
arr = [1, 2, 3]
m = len(arr)
print(count(arr, m, 4))
'''
1. Given a sorted circularly linked list of Nodes that store 
integers and a new Node, 
insert the new Node into the correct position. (Duplicates allowed)  
2. longest substring
3. reverse a string using recursion
4. reverse a string using 1 for loop
5. reverse a string using a stack
'''

#reverse a number:
def reverseNumber(number):
    total =0
    while number > 0:
        digit = number % 10
        number = number / 10
        total = total * 10 + digit
    return total

#find 2 sum
def find2sum(array, total):
	s = {}

	for i in array:
		remaining = total - i
		if remaining in  s:
			print remaining, s[remaining]
		s[remaining] = i


#check whether the tree is BST
class TreeNode:
	def __init__(self,data):
		self.data = data
		self.left = None
		self.right = None

root = TreeNode(50)
root.left = TreeNode(5)
root.right = TreeNode(20)
root.left.left = 1
root.left.right = 7
root.right.left = 15
root.right.right = 25

def checkBST(root):
	if root is None:
		return True
	if root.left.data < root.data and root.right.data > root.data:
		return True
	else: 
		return False
	return checkBST(root.left) and checkBST(root.right)

#doubly linked list implementation
class DNode:
	def __init__(self,data,nextNode, prevNode):
		self.nextNode = nextNode
		self.prevNode = prevNode
		self.data = data

class DoublyLinkedList:
	def __init__(self,node):
		self.first = node
		self.last = node

	def append(self,node):
		node.next = self.first
		self.first.prev = node
		self.first.next = None
		self.first = node

	def pop(self):
		toBePop = self.last
		toBeTail = self.last.prev
		toBeTail.next = None
		toBePop.prev = None
		self.last = toBeTail
		return toBePop


#dijkistra 
import heapq


def calculate_distances(graph, starting_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[starting_vertex] = 0
    pq = []

    for vertex, distance in distances.items():
        # entry = [distance, vertex]
        heapq.heappush(pq, entry)

    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)
        for neighbor, neighbor_distance in graph[current_vertex].items():
            distance = distances[current_vertex] + neighbor_distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
    return distances


example_graph = {
    'U': {'V': 2, 'W': 5, 'X': 1},
    'V': {'U': 2, 'X': 2, 'W': 3},
    'W': {'V': 3, 'U': 5, 'X': 3, 'Y': 1, 'Z': 5},
    'X': {'U': 1, 'V': 2, 'W': 3, 'Y': 1},
    'Y': {'X': 1, 'W': 1, 'Z': 1},
    'Z': {'W': 5, 'Y': 1},
}
print calculate_distances(example_graph, 'X')
# => {'U': 1, 'W': 2, 'V': 2, 'Y': 1, 'X': 0, 'Z': 2}


#divide 2 numbers without using division:
def divide(dividend,divisor):
    if dividend < 0:
        return -1
    return 1 + divide(dividend - divisor,divisor)

print divide(7,3)
print divide(9,3)
print divide(10,3)

#do ksum problem for 2 lists, amz interview problems
def find2sumInArrayEvenTarget(array,target):
	storage = {}

	for i in array:
		if i in storage:
			return i, storage[i]
		storage[target - i] = i
	return False

#find lowest distances to all of values in an array


#change all surrounded 0 to 1, leetcode 130:
graph = [[1 for _ in range(4)] for _ in range(4)]
graph[1][1] = 0
graph[1][2] = 0
graph[2][2] = 0
graph[3][1] = 0
for i in graph:
	print i

def surroundedRegions(graph):
	# tracking = [[False for i in range(4)] for i in range(4)]
	rows = len(graph)
	columns = len(graph[0])
	changeTo1 = []
	for row in rows:
		for column in columns:
			if graph[row][column] == 0:
				DFS(graph,row,column,changeTo1)

def BFS(graph,row,column,changeTo1):
	neighbors = [[1,1],[1,-1],[-1,1],[-1,-1]]

	for neighbor in neighbors:
		nextNeighbor = graph[row + neighbor[0]] + graph[column + neighbor[1]]
		if nextNeighbor == 0 and not border case and not visited:
			BFS(graph,row+neighbor[0],column + neighbor[1])
			changeTo1.append()


#edit distances
# Given two strings str1 and str2 and below operations that can performed on str1.
# Find minimum number of edits (operations) required to convert ‘str1’ into ‘str2’.
# Insert
# Remove
# Replace
def editDistance(str1,str2,m,n):
	#if the first string is empty, return the entire other string
	if m == 0 :
		return n
	if n == 0:
		return m
	#if last characters of 2 strings are the same, move on. get sub string of the 
	#rest of the string
	if str1[m-1] == str2[n-1]:
		return editDistance(str1,str2,m-1,n-1)

	return 1 + min(editDistance(str1,str2,m, n-1), #insert
					editDistance(str1,str2,m - 1, n) #remove
					editDistance(str1,str2,m-1,n-1) #replace
		)
def editDistDP(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):
 
            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
 
    return dp[m][n]
# Driver program
str1 = "sunday"
str2 = "saturday"
editDistDP(str1, str2, len(str1), len(str2))

#leetcode 91 : make combination of "123" => [1,23], [1,2,3], [12,3]


#leetcode word ladder:
def ladderLength(beginWord,endWord,wordList):
	wordList.add(endWord)
	cur_level = [beginWord]
	next_level = []
	depth = 1
	n = len(beginWord)
	while cur_level:
		for item in cur_level : 
			if item == endWord:
				return depth
			for i in range(n):
				for c in 'abcdefghijklmnopqrstuvwxyz':
					word = item[:i] + c +item[i +1 :]
					if word in wordList:
						wordList.remove(word)
						next_level.append(word)
		depth += 1
		cur_level = next_level
		next_level = 0
	return 0
ladderLength("hit", "cog", {"hot", "dot", "dog", "lot", "log"}) == 5

#longest substring without repeating character
def lengthOfLongestSubString(s):
	if not s:
		return 0
	if len(s) <= 1:
		return len(s)
	locations = [-1 for i in range(256)]
	index = -1
	m = 0
	for i,v in enumerate(s):
		if locations[ord(v)] > index:
			index = locations[ord(v)]
		m = max(m, i - index)
		locations[ord(v)] = 1
	return m

#There are 2 sorted arrays A and B of size n each.
# Write an algorithm to find the median of the array 
#obtained after merging the above 2 arrays(i.e. array of length 2n).
# The complexity should be O(log(n)).
def findMedianSortedArrays(nums1, nums2):
	length1 = len(nums1)
	length2 = len(nums2)
	k = (length1 + length2) / 2
	if (length1 + length2) % 2 == 0 :
		return (findK(nums1,num2,k) + findK(nums1,num2, k -1)) / 2.0
	else:
		return findK(nums1,nums2, k)

def findK(num1,num2,k):
	if not num1:
		return num2[k]
	if not num2:
		return num1[k]
	if k == 0:
		return min(num1[0], num2[0])

	length1 = len(num1)
	length2 = len(num2)
	if num1[length1 / 2] > num2[length2 / 2]:
		if k > length1 / 2 + length2 / 2:
			return findK(num1, num2[length2 /2 + 1:], k - length2 / 2 - 1)
		else:
			return findK(num1[:length1 / 2], num2, k)
	else:
		if k > length1 / 2 + length2 / 2:
			return findK(num1[length1 / 2 + 1:], num2 , k - length1 /2 -1)
		else:
			return findK(num1,num2[:length / 2], k)


#longest palondromic substring:
def longestPalSubstr(string):
	maxLength = 1
	start = 0
	length = len(string)
	low = 0
	high = 0

	#one by one consider every character as center point of even
	#and length palindromes
	for i in range(1,length):
		low = i - 1
		high = i
		while low >= 0 and high < length and string[low] == string[high]:
			if high - low + 1 > maxLength:
				start = low
				maxLength = high - low + 1
			low -= 1
			high += 1

		low  = i - 1
		high = i + 1
		while low >= 0 and high < length and string[low]== string[high]:
			if high - low + 1 > maxLength:
				start = low
				maxLength = high - low + 1
			low -= 1
			high += 1
	return maxLength
print longestPalSubstr('forgeeksskeegfor')


#matching wild card
def match(first,second):
	#if we reach the end of both string, we are done
	if len(first) == 0 and len(second) == 0:
		return True

	#make sure that the characters after '*' are present
	#in second string. This function assumes that the first
	#string will not contain two consecutive '*'
	if len(first) > 1 and first[0] == '*' and len(second) == 0:
		return False

	#if the string contains '?' or current characters of both
	#strings match
	if(len(first) > 1 and first[0] == '?') or (len(first) != 0) and len(second) != 0 and first[0] == second[0]:
		return match(first[1:], second[1:])

	#if there is *, then there are two possibilities
	# a/ we consider current character of second string
	# b/ we ignore current character of second string
	if len(first) != 0 and first[0] == '*':
		return match(first[1:], second) or match(first, second[1:])
	return False

def test(first,second):
	if match(first,second):
		print "yes"
	else:
		print "no"

#Roman to integer conversion
def romanToInt(s):
	map = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
	result = 0
	for i in range(len(s)):
		if i > 0 and map[s[i]] > map[s[i-1]]:
			result -= map[s[i-1]]
			result += map[s[i]] - map[s[i-1]]
		else:
			result += map[s[i]]
	return result
print romanToInt("XII") 

#longest common prefix
def longestCommonPrefix(word1,word2):
	pointer 0
	count = 0 
	while pointer< len(word1) and pointer < len(word2):
		if word1[pointer] != word2[pointer]:
			return count
		count += 1
		pointer += 1
	return count

#3Sum problem
def threeSum(nums):
	nums.sort()
	result = []
	i = 0
	while i <len(nums) - 2:
		j = i + 1
		k = len(nums) - 1
		while j < k :
			l = [nums[i], nums[j], nums[k]]
			if sum(l) == 0:
				result.append(l)
				j += 1
				k -= 1
				while j < k and nums[j] == nums[j-1]:
					 j += 1
				while j < k and nums[k] == nums[k + 1]:
					k -= 1
			elif sum(l) > 0:
				k -= 1
			else:
				j+= 1
		i += 1
		#ignore repeat numbers
		while i < len(nums) -2 and nums[i] == nums[i-1]:
			i += 1
	return result

print threeSum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]

#letter combinations of a phone numbers:
digit2letters = {
        '2': "abc",
        '3': "def",
        '4': "ghi",
        '5': "jkl",
        '6': "mno",
        '7': "pqrs",
        '8': "tuv",
        '9': "wxyz",
    }
def letterCombinations(digits):
	if not digits:
		return []
	result = []
	dfs(digits, "", result)
	return result
def dfs(digits,current,result):
	if not digits:
		result.append(current)
		return
	for c in digit2letters[digits[0]]:
		dfs(digits[1:], current + c, result)
#testing
letterCombinations('23') == ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]


## merge 2 sorted linked list
def merge2linkedList(l1,l2):
	pass

l1 = LinkedList()
l2 = LinkedList()
l1.add(1)
l1.add(3)
l1.add(6)
l2.add(2)
l2.add(4)
l2.add(7)
#expect 1,2,3,4,6,7


#Given an array nums and a value val, remove all instances of 
#that value in-place and return the new length.
# Given nums = [3,2,2,3], val = 3,
# Your function should return length = 2, with the first two elements of nums being 2.
def removeElement(nums,val):
	left = 0 
	right = len(nums) - 1
	while left <= right:
		while left <= right and num[left] != val:
			left += 1
		while left <= right and num[right] == val:
			right -= 1
		if left < right:
			nums[left] = num[right]
			left += 1
			right -= 1
	return right + 1
removeElement([1, 2, 3, 4, 3, 2, 1], 1) == 5
removeElement([2], 3) == 1



#LRU implementation with circularly linkedlist and hashmap
#replace,insert, delete must be 1.
'''
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.
get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.
'''
class LRUCache:
	class Node:
		def __init__(self,key,value):
			self.key = key
			self.value = value
			self.prev, self.next = None, None

	def __init__(self,capacity):
		self.capacity, self.size = capacity, 0
		self.dic = {}
		self.head, self.tail = self.Node(-1,-1), self.Node(-1, -1)
		self.head.next = self.tail
		self.tail.prev = self.head

	def __remove(self,node):
		node.prev.next = node.next
		node.next.prev = node.prev
		node.prev, node.next = None, None

	def __insert(self,node):
		node.prev = self.head
		node.next = self.head.next
		self.head.next.prev = node
		self.head.next = node

	def get(self,key):
		if key not in self.dic:
			return -1
		node = self.dic[key]
		self.__remove(node)
		self.__insert(node)
		return node.value

	def set(self,key,value):
		if key in self.dic:
			node = self.dic[key]
			self.__remove(node)
			node.value = value
			self.__insert(node)
		else:
			if self.size == self.capacity:
				discard = self.tail.prev
				self.__remove(discard)
				del self.dic[discard.key]
				self.size -= 1
			node = self.Node(key,value)
			self.dic[key] = node
			self.__insert(node)
			self.size += 1

#test cases:
lru_cache = LRUCache(3)
    lru_cache.set(1, 1)
    lru_cache.set(2, 2)
    lru_cache.set(3, 3)
    assert lru_cache.get(0) == -1
    assert lru_cache.get(1) == 1
    lru_cache.set(1, 10)
    assert lru_cache.get(1) == 10
    lru_cache.set(4, 4)
    assert lru_cache.get(2) == -1

#==============DP problem set=================================
"""
Problem Statement
=================
Find the length of the longest Bitonic Sequence in a given sequence of numbers. A Bitonic sequence is a sequence of
numbers which are increasing and then decreasing."""

def longest_bitonic(sequence):
	length_of_input = len(sequence)
	increasing_sequence = [1] * length_of_input
	decreasing_sequence = [1] * length_of_input

	for i in range(1,length_of_input):
		for j in range(0,i):
			if sequence[i] > sequence[j]:
				decreasing_sequence[i] = max(increasing_sequence[i], increasing_sequence[j] + 1)

	for i in range(length_of_input - 2, -1, -1):
		for j in range(length_of_input - 1, i, -1):
			if sequence[i] > sequence[j]:
				decreasing_sequence[i] = max(decreasing_sequence[i], decreasing_sequence[j] + 1)

	max_value = 0

	for i in range(len(sequence)):
		bitonic_sequence_length = increasing_sequence[i] + decreasing_sequence[i] - 1
		max_value = max(max_value,bitonic_sequence_length)

	return max_value

longest_bitonic([1,4,3,7,2,1,8,11,13,0]) == 7 # 1, 4, 7, 8, 11, 13, 0


#decode ways:
# Input:  digits[] = "121"
# Output: 3
# // The possible decodings are "ABA", "AU", "LA"

# Input: digits[] = "1234"
# Output: 3
# // The possible decodings are "ABCD", "LCD", "AWD"
def countDecodingDP(digits,n):
	#a table to store results of subproblems
	count = [0] * (n+1)
	count[0] = 1
	count[1] = 1

	for i in range(2,n+1):
		count[i] = 0

		#if the last digit is not 0,
		#then last digit must add to the number of words
		if digits[i-1] > 0:
			count[i] = count[i-1]

		#if second last digit is smaller than 2 and last digit is smaller than 7
		#then last two digits from a valid character
		if digits[i - 2] == 1 or (digits[i-2] == 2 and digit[i-1] < 7):
			count[i] += count[i-2]
	return count[n]
#test program
digits = [1,2,3,4]
n = len(digits)

print "Count is ", countDecodingDP(digits,n)


#coin changes number of ways:
'''
Problem: given a total and coins of certain denominations find number of ways
total can be formed from coins assuming infinity supply of coins
'''
def min_coins(coins, total):
    cols = total + 1
    rows = len(coins)
    T = [[0 if col == 0 else float("inf") for col in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(1, cols):
            if j < coins[i]:
                T[i][j] = T[i - 1][j]
            else:
                T[i][j] = min(T[i - 1][j], 1 + T[i][j - coins[i]])
    for i in T:
    	print i
    return T[rows - 1][cols - 1]

coins = [1,2,3]
total = 5
# assert expected == coin_changing_num_ways(coins, total)
min_coins(coins, total)


#fibbonaci recursive and DP
#recursive : O(2^N)
def fibonacci_recursive(n):
	if n == 0 or n == 1:
		return n
	return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

#DP is O(n)
def fibonacci(n):
	n1,n2 = 0,1
	if n == n1 or n == n2:
		return n

	for i in range(2, n + 1):
		n1, n2 = n2, n1 + n2
	return n2
assert 610 == fibonacci_recursive(15)
assert 610 == fibonacci(15)


#longest common substring
"""
Problem Statement
=================
Given two sequences A = [A1, A2, A3,..., An] and B = [B1, B2, B3,..., Bm], find the length of the longest common
substring.
Complexity
----------
* Recursive Solution: O(2^n) (or O(2^m) whichever of n and m is larger).
* Dynamic Programming Solution: O(n * m)
"""

#recursive solution
def longest_common_string_recursive_helper(str1,str2,pos1,pos2,check_equal):
	if pos1 == -1 or pos2 == -1:
		return 0

	if check_equal:
		if str1[pos1] == str2[pos2]:
			return 1 + longest_common_string_recursive_helper(str1,str2,pos1 - 1, pos2 - 1, True)
		else:
			return 0

	longest = 0 #start (again) to find the longest from the current positions

	if str1[pos1] == str2[pos2]:
		longest = 1 + longest_common_string_recursive_helper(str1,str2,pos1 -1, pos2 -1, True)
	return max(longest,
               longest_common_string_recursive_helper(str1, str2, pos1, pos2 - 1, False),
               longest_common_string_recursive_helper(str1, str2, pos1 - 1, pos2, False))

def longest_common_substring_recursive(str1, str2):
    return longest_common_string_recursive_helper(str1, str2, len(str1) - 1, len(str2) - 1, False)

#dynamic programming solution:
def longest_common_substring(str1, str2):
    cols = len(str1) + 1     # Add 1 to represent 0 valued col for DP
    rows = len(str2) + 1     # Add 1 to represent 0 valued row for DP

    T = [[0 for _ in range(cols)] for _ in range(rows)]

    max_length = 0

    for i in range(1, rows):
        for j in range(1, cols):
            if str2[i - 1] == str1[j - 1]:
                T[i][j] = T[i - 1][j - 1] + 1
                max_length = max(max_length, T[i][j])

    return max_length

#test cases
str1 = "abcdef"
str2 = "zcdemf"
expected = 3
assert expected == longest_common_substring(str1, str2)
assert expected == longest_common_substring_recursive(str1, str2)


#cutting rod for sale
'''given a rod of length n inches and an array of prices that contains prices
of all pieces of size smaller than n. Determine the maximum value obtainable
by cutting up the rod and selling the pieces

Recursive solution = O(2^n)
Dynamic programming solution = O(n^2)'''
def max_profit_recursive(prices, rod_length):
	if rod_length == 0 :
		return 0

	max_price = float('-inf')

	for length in range(1, rod_length + 1):
		max_price = max(max_price, prices[length - 1] + max_profit_recursive(prices, rod_length - length))

	return max_price

def max_profit_dp(prices,rod_length):
	rod_length_values = [0 for _ in range(rod_length + 1)]

	for length in range(1, rod_length + 1):
		max_value = float("-inf")
		for cut_length in range(1, length + 1):
			max_value = max(max_value, prices[cut_length - 1] + rod_length_values[length - cut_length])
		rod_length_values[length] = max_value
	return rod_length_values[rod_length]

prices = [3,5,8,9,10,20,22,25]
rod_length = 8
expected_max_profit = 26
assert expected_max_profit == max_profit_recursive(prices, rod_length)
assert expected_max_profit == max_profit_dp(prices, rod_length)


#find kth ugly number
'''ugly numbers are numbers whose only prime factors are 2,3, or 5. the sequence
1,2,3,5,6,8,9,10,12,15 shows the first 11 ugly number

write the program to find the kth ugly number
Time complexity O(n)
Space Complexity O(n)
'''

def ugly_number(kth):
	ugly_factors = [1] #by convention 1 is included
	factor_index = {
		2:0,
		3:0,
		5:0
	}

	for num in range(1,kth):
		minimal_factor = min(min(ugly_factors[factor_index[2]] * 2, ugly_factors[factor_index[3]]*3), ugly_factors[factor_index[5]] * 5)
		ugly_factors.append(minimal_factor)

		for factor in [2,3,5]:
			if minimal_factor % factor == 0:
				factor_index[factor] += 1
	return ugly_factors[kth - 1]
#test case
assert 5832 == ugly_number(150)


#count number of binary search trees created for array of size n. The solution
#is the nth catalan number
#Complexity:
# Dynamic Programing: O(n^2)
#Recursive solution: O(2^n)
def num_bst(num_nodes):
	T = [0 for _ in range(num_nodes + 1)]
	T[0] = 1
	T[1] = 1

	for node in range(2, num_nodes + 1):
		for sub in range(0,node):
			T[node] += T[sub] * T[node - sub - 1]

	return T[num_nodes]

def num_bst_recursive(num_nodes):
	if num_nodes == 0 or num_nodes == 1:
		return 1

	result = 0

	for root in range(1,num_nodes + 1):
		result += num_bst_recursive(root - 1) * num_bst_recursive(num_nodes - root)

	return result
#test cases
assert 5 == num_bst(3)
assert 5 == num_bst_recursive(3)


#longest increasing subsequent

def longestIncreasingSubsequent(arr):
    n = len(arr)
 
    # Declare the list (array) for LIS and initialize LIS
    # values for all indexes
    lis = [1]*n
 
    # Compute optimized LIS values in bottom up manner
    for i in range (1 , n):
        for j in range(0 , i):
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 :
                lis[i] = lis[j]+1
 
    return max(lis)
# end of lis function


#count the number of paths from 1,1 to N,M in an NxM matrix
#Dynamic programming: O(M x N)
#recursive: O(2^cols) if cols > rows else O(2^rows)

def num_paths_matrix(rows,cols):
	T = [ [1 if row == 0 or col == 0 else 0 for row in range(cols)] for col in range(rows)]
	for row in range(1,rows):
		for col in range(1,cols):
			T[row][col] = T[row - 1][col] + T[row][col - 1]

	return T[rows - 1][cols - 1]

def num_paths_matrix_recursive(rows,cols):
	if rows == 1 or cols == 1:
		return 1
	return num_paths_matrix_recursive(rows - 1, cols) + num_paths_matrix_recursive(rows,cols - 1)


# A robot is located at the top-left corner of a m x n grid 
# (marked 'Start' in the diagram below).
# The robot can only move either down or right at any point in time. 
# The robot is trying to reach the
#  bottom-right corner of the grid (marked 'Finish' in the diagram below).
# How many possible unique paths are there?
# Input: m = 3, n = 2
# Output: 3

def uniquePaths(m,n):
	dp = [ [1 for _ in range(n)] for _ in range(m)]
	for row in range(1,n):
		for col in range(1,m):
			dp[col][row] = dp[col - 1][row] + dp[col][row - 1]
	return dp[m-1][n-1]

assert uniquePaths(3,7) == 28


#===============end DP problems ============================
'''
Given a sorted array of integers, find the starting and ending position of a given target value.
Your algorithm's runtime complexity must be in the order of O(log n).
If the target is not found in the array, return [-1, -1].
For example,
Given [5, 7, 7, 8, 8, 10] and target value 8,
return [3, 4].
'''
def searchRange(nums,target):
	result = []
	length = len(nums)
	start = 0
	end = length

	while start < end:
		mid = (start + end ) /2
		if nums[mid] == target and ( mid == 0 or nums[mid - 1] != target):
			result.append(mid)
			break
		if nums[mid] < target:
			start = mid + 1
		else:
			end = mid
	if not result:
		return [-1, -1]

	end = length
	while start < end:
		mid = (start + end) / 2
		if nums[mid] == target and (mid == length - 1 or nums[mid + 1] != target):
			result.append(mid)
			break
		if nums[mid] <= target:
			start = mid + 1
		else:
			end = mid
	return result
#test cases
assert searchRange([5, 7, 7, 8, 8, 10], 8) == [3, 4]


#remove duplicate in sorted array
def removeDuplicates(nums):
	if not nums:
		return 0

	#the index where the character needs to be placed
	index = 1
	#the index of repeating characters
	start = 0
	for i in range(1, len(nums)):
		if nums[start] != nums[i]:
			nums[index] = nums[i]
			index += 1
			start = 1
	return index
#test cases
assert removeDuplicates([1,1,2]) == 2 # array after remove duplicates is [1,2] which is length == 2

#palindrome partition:
'''
Given a string s, partition s such that every substring of the partition is a palindrome.
Return all possible palindrome partitioning of s.
For example, given s = "aab",
Return
  [
    ["aa","b"],
    ["a","a","b"]
  ]
'''
def partition(s):
	if not s:
		return [[]]
	result = []
	for i in range(len(s)):
		if self.isPalindrome(s[:i + 1]):
			for r in partition(s[i+1:]):
				result.append([s[:i+1]] + r)
	return result

def isPalindrome(s):
	return s == s[::-1]


#minimum path sum
'''
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.
'''









