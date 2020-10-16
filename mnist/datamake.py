import sys
import csv

def csv_to_in(csv_path):
    index = 0
    with open(csv_path, 'r') as csv_file:
        datas = csv.reader(csv_file, delimiter=',')
        for data in datas:
            # print(data)
            label = int(data[0])
            print(label)
            data = data[1:]
            assert (len(data) == 28 * 28)
            with open("mnist/mnist_{}_local_property.in".format(index),"w") as single_file:
                for i in range(28):
                    for j in range(28):
                        # single_file.write(str(round(float(data[i*28+j])*255)))
                        single_file.write(data[i*28+j])
                        single_file.write("\n")
                        # single_file.write("\n" if j == 27 else " ")
                for k in range(10):
                    if k==label:
                        continue
                    property_list=[0]*11
                    property_list[label]=-1
                    property_list[k]=1
                    single_file.write(str(property_list)[1:-1].replace(',',''))
                    single_file.write('\n')                        
                index = index + 1
            if index==100:
                break

def main(argv):
    assert(len(argv) == 1)
    csv_to_in(argv[0])

if __name__ == "__main__":
    # main(sys.argv[1:])
    main(['mnist/mnist_test.csv'])
