import pickle

# open a file, where you stored the pickled data
file = open('./file.pkl', 'rb')

# dump information to that file
data = pickle.load(file)
print(len(data))
# close the file
file.close()
