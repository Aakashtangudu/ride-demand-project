with open('data/indore_ola.csv', encoding='utf-8') as f:
 lines = f.readlines()

cleaned = []
for line in lines:
 line = line.strip().strip(chr(34))
 cleaned.append(line + chr(10))

with open('data/indore_ola_clean.csv', 'w', encoding='utf-8') as f:
 f.writelines(cleaned)

print('Done')