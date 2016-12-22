import pickle


users = {'kim':'3kid9', 'sun80':'393948', 'ljm':'py90390'}
f = open('users.txt', 'w')
pickle.dump(users, f)
f.close();

f = open('/home/bbu/Workspace/data/result/GTEA/DB.pkl');
a = pickle.load(f)

print a.opts