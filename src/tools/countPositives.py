outputPath = r'C:\Users\lykos\Desktop\py-mgr\src\tools\data.txt'

with open(outputPath, 'r') as f:
    lines = f.readlines()

counter = 0
for line in lines:
    parts = line.split('.jpg')
    count = parts[1].strip().split(' ')[0]
    counter += int(count)


print('Total positives probes', counter)