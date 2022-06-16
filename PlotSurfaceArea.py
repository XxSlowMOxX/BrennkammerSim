def shitplot(arr, plotheight = 20):
    maxval = max(arr)
    step = maxval / plotheight
    valheights = []
    for val in arr:
        valheights.append(round(val / step))
    print(valheights)
    print("PLOT:")
    for row in range(plotheight):
        height = round((maxval - (step * row)))
        rowstr = str((height)).ljust(len(str(maxval))) + ":"
        for i in range(len(arr)):
            #rowstr += "|" + str(valheights[i]) + ">" + str(height)
            if (valheights[i] > (plotheight - row)):
                rowstr += "#"
            elif (valheights[i] == (plotheight - row)):
                rowstr += "*"
            else:
                rowstr += " "
        print(rowstr)
        
intarr = []
with open("Simulation\\Surface.txt", "r") as surfacefile:
    normarr = surfacefile.read().split(";")
    for string in normarr[:-1]:
        intarr.append(int(string))
    shitplot(intarr, 20)
