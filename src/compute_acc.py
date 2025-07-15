import json

with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)

latent_map = data['results']
cnt_valid = 0
cnt_correct = 0
for latent in latent_map:
    if latent_map[latent]['score'] is None:
        continue

    if latent_map[latent]['score'] == 0:
        continue
    
    if latent_map[latent]['score'] * latent_map[latent]['weight'] > 0:
        cnt_correct += 1
    cnt_valid += 1 

print(cnt_correct, cnt_valid)
print(cnt_correct/cnt_valid)