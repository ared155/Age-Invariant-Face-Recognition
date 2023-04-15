import csv
let='D'
r=open('USER_DETAILS.csv', mode = 'r')
rr=r.readlines()
for row in rr:
    r1,r2,r3,r4=row.split(",")
    print(r2)
    if let in r2:
        print(r4)
##        
####    print("r1 {} \n  r2 {} \n  r3 {} \n  r4 {} \n let {} \n".format(r1,r2,r3,r4,let))
##    if str(r2) ==str(let):
##        print("ssss")
print("hi")
##with open('USER_DETAILS.csv', mode = 'r') as file:
##                                print("Entered")
####                                csvFile  = file.readlines()
##                                csvFile=open()
##                                csvFile = csv.reader(file)
####                                print(csvFile)
##                                for index,row in enumerate(csvFile) :
##                                    print(row)
####                                    if index == 2:
####                                        print(row[1])
######                                    if let in row:
######                                            print(row[0])
####                                            print(row[1])
####                                            print(row[2])
####                                            print(row[3])
####                                            print("AGE is {}".format(row[1]))
####                                            print("NAME is {}".format(row[2]))
####                                            print("CRIME is {}".format(row[3]))
####                                            
####
