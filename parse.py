def parse_attack_stat_1px(path, mode='c'):
    file = open(path, 'r', encoding='UTF-8')
    text = file.read()
    file.close()

    lines = text.split('\n')
    lines.remove('')
    lines = [line.split(',') for line in lines]
    data = {}
    for line in lines[1::]:
        data[line[0]] = {'cln_label': int(line[1]), 'adv_label': int(line[2]), 'pixels': []}
        if mode == 'c':
            data[line[0]]['pixels'].append(
                {'position': (int(line[3]), int(line[4])), 'adv_color': (int(line[5]), int(line[6]), int(line[7])),
                 'cln_color': (int(line[8]), int(line[9]), int(line[10]))})
        elif True: #mode == 'g' or mode == 'bw' or mode == 'bwz':
            data[line[0]]['pixels'].append(
                {'position': (int(line[3]), int(line[4])), 'adv_color': (int(line[5])),
                 'cln_color': (int(line[6]))})
        else:
            raise BaseException('WRONG MODE - parse_attack_stat')

    return data


def parse_attack_stat_jsma(path, mode='c'):
    file = open(path, 'r', encoding='UTF-8')
    text = file.read()
    file.close()

    data = {}
    lines = text.split('\n')
    while '' in lines:
        lines.remove('')
    for line in lines[1::]:
        d = {'cln_label': int(line.split(',')[1]), 'adv_label': int(line.split(',')[2]), 'pixels': []}

        try:
            x = line.split(',')[3].split('■')
            while '' in x:
                x.remove('')
            for el in x:
                meta = []
                sels = el.split('|')
                for part in sels:
                    spart = part.split(';')
                    s = []
                    for x in spart:
                        s.append(int(x))
                    meta.append(s)

                if mode == 'c':
                    d['pixels'].append(
                        {'position': (meta[0][0], meta[0][1]), 'adv_color': (meta[1][0], meta[1][1], meta[1][2]),
                         'cln_color': (meta[2][0], meta[2][1], meta[2][2])})
                else:
                    d['pixels'].append(
                        {'position': (meta[0][0], meta[0][1]), 'adv_color': (meta[1][0]), 'cln_color': (meta[2][0])})

            if d['pixels']:
                data[line.split(',')[0]] = d

        except BaseException:
            continue

    return data
