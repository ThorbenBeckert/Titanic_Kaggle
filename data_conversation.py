# the program converts two csv-files filled with str, float and int into a new file filled with just floats and integers.
# all not existing data will get filled with a 0. This could be a problem with the age of some people, but should be ok.
# For the neural network not exisiting data should be a problem, if they are not filled systematically. This will be the next step after trying some setups of NN.


import os

import pandas as pd
import numpy as np
def convert_data_titanic(pathtodata):
    testdataframe = pd.read_csv(pathtodata)
    # data contains: 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

    
    passengerid = testdataframe.get('PassengerId').to_numpy().astype(np.int16)

    # try if 'Survived' exist. If so build '*Y.dat' as new file with PassengerId,Survived (1/0) for octave/matlab. Y = output layer of the neural network
    try:
        testdataframe.Survived.any()==1
        datafilenewY = pathtodata[:-4]+'Y.dat'
        with open(datafilenewY, 'w') as handle:
            handle.truncate(0)
            survived = testdataframe.get('Survived').to_numpy().astype(np.int16)
            for i in range(len(passengerid)):
                handle.write('%i,%i\n' % (passengerid[i], survived[i]))
    except:
        #should trigger within test data...
        pass
    
    finally:
        #do the conversation and save of numeric data in '*X.dat'. Y = input layer of the neural network
        datafilenewX = pathtodata[:-4]+'X.dat'


        
        #We ignore Name and Ticket and convert the rest
        pclass = testdataframe.get('Pclass').fillna(0).to_numpy().astype(np.int16)

        sex = testdataframe.get('Sex').fillna(0)
        for i in range(len(sex)):
            if sex[i] == "female":
                sex[i] = 1
            elif sex[i] == "male":
                sex[i] = 2
            elif sex[i] == 0:
                pass
            else:
                print("%i has no sex?" % passengerid[i])
        sex = sex.to_numpy().astype(np.int16)

        age = testdataframe.get('Age').fillna(0).to_numpy().astype(np.float64)
        
        sibsp = testdataframe.get('SibSp').fillna(0).to_numpy().astype(np.int16)
        
        parch = testdataframe.get('Parch').fillna(0).to_numpy().astype(np.int16)
        
        fare = testdataframe.get('Fare').fillna(0).to_numpy().astype(np.float64)

        cabin = testdataframe.get('Cabin').fillna('NONE')
        cabin_list = []
        for c in cabin.to_list():
            if c.startswith('A'):
                cabin_list.append(1)
            elif c.startswith('B'):
                cabin_list.append(2)
            elif c.startswith('C'):
                cabin_list.append(3)
            elif c.startswith('D'):
                cabin_list.append(4)
            elif c.startswith('E'):
                cabin_list.append(5)
            elif c.startswith('F'):
                cabin_list.append(6)
            elif c.startswith('G'):
                cabin_list.append(7)
            elif c == "NONE":
                cabin_list.append(0)
            else:
                cabin_list.append(-1)
                print('Weird cabin string: {} Mapped to 0'.format(c))
        cabin = np.array(cabin_list, dtype=np.int16)
        
        embarked = testdataframe.get('Embarked').fillna('NONE')
        for i in range(len(embarked)):
            if embarked[i] == "S":
                embarked[i] = 1
            elif embarked[i] == "C":
                embarked[i] = 2
            elif embarked[i] == "Q":
                embarked[i] = 3
            elif embarked[i] == "NONE":
                embarked[i] = 0
            else:
                print("%i suddenly ported to the titanic?" % passengerid[i])
                embarked[i] = -1
        embarked = embarked.to_numpy().astype(np.int16)



        #save Data in new File X
        with open(datafilenewX, 'w') as handle:
            handle.truncate(0)
            for i in range(len(passengerid)):
                handle.write('%i,%i,%i,%f,%i,%i,%f,%i,%i\n' % (passengerid[i], pclass[i], sex[i], age[i], sibsp[i], parch[i], fare[i], cabin[i], embarked[i]))


convert_data_titanic("input/train.csv")
convert_data_titanic("input/test.csv")
