


for instance in ['bert_inter.a', 'bert_itermax.a', 'bert_mwmf.a']:
    out_file = open(instance, 'w', encoding='utf-8')
    for i in range(5):
        file_path = "fold_" + str(i)
        with open(file_path, 'r', encoding='utf-8') as in_file:
            for x in in_file:
                out_file.write(x)
        print(file_path)
    print('=='*50)















