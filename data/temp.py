start = 640.

for i in range(640):
    current = (486./640.)*start
    if current % 1 == 0:
        print('yo')
        print(str(current) + ' ' + str(start))
    start -= 1
