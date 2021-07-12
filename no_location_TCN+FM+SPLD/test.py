max = 0
with open("./neg_url_tokens/domain_exist_at_nec_tokens.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()[:373946]
    for line in lines:
        line_list = line.split()
        if len(line_list) > max:
            max = len(line_list)

print(max)
# max = 0
# with open("./neg_url_tokens/domain_exist_at_nec_tokens.txt", mode='r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print(len(lines))
