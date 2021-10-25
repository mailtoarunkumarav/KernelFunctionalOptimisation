import sys
import datetime
import os


class PrintHelper:

    def __init__(self, directory):
        original_stdout = sys.stdout
        timenow = datetime.datetime.now()
        stamp = timenow.strftime("%H%M%S_%d%m%Y")
        # TT : PrintFile printConsole
        PrintHelper.p1 = "TT"
        PrintHelper.p2 = "TF"
        PrintHelper.p3 = "FT"
        PrintHelper.p4 = "FF"
        PrintHelper.console_output = open(directory+'/console_output_' + str(stamp) + '.txt', 'w')

    @staticmethod
    def printme(*args):

        if type(args[0]) == str and len(args[0])== 2:

            # If true print it to the console
            if args[0][1] == 'T':
                for each1 in args[1:]:
                    print(each1, end=' ')
                print("")
            # If true print it to file
            if args[0][0] == 'T':
                for each2 in args[1:]:
                    PrintHelper.console_output.write(str(each2) + " ")
                PrintHelper.console_output.write("\n")
                PrintHelper.console_output.flush()