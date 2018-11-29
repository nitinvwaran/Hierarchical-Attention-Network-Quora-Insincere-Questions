import pandas as pd


def process_file(file_name):

    out_file = '/home/nitin/Downloads/nrma_motor_parsed_out.txt'
    with open (out_file,'w') as wf:
        with open(file_name,'r') as f:
            for line in f:
                l = line.replace('\n','').replace('*','')
                wf.write(l + '\n')




def main():

    f = '/home/nitin/Downloads/nrma_motor_parsed.txt'

    process_file(f)


main()