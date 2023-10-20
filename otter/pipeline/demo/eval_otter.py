import json
import copy

def main():
    set_true, set_false, set_cnt = 0, 0, 0
    c2_true, c2_cnt, c3_true, c3_cnt, c4_true, c4_cnt, c5_true, c5_cnt = 0, 0, 0, 0, 0, 0, 0, 0
    non_cs_true, cs_true, cs_cnt, non_cs_cnt= 0, 0, 0, 0
    pos_true, pos_cnt, neg_true, neg_cnt = 0
    ground_truth, prediction = [], []
    cs_true_list, cs_cnt_list = [0]*10, [0]*10

    results_path = 'otter_bow.json'

    with open(results_path, 'r') as f:
        results = json.load(f)

        for result in results:
            uid = result['uid']
            commonSense = result['commonSense']
            concept = result['concept']
            c_len = len(concept.split(' '))

            # answer
            answer = result['answer']
            
            if uid[-1] == ('A' if answer == 'positive' else 'B'):
                set_true += 1

                if c_len == 2:
                    c2_true += 1
                    c2_cnt += 1
                if c_len == 3:
                    c3_true += 1
                    c3_cnt += 1
                if c_len == 4:
                    c4_true += 1
                    c4_cnt += 1
                if c_len == 5:
                    c5_true += 1
                    c5_cnt += 1

                if commonSense == '0':
                    non_cs_true += 1
                    non_cs_cnt += 1
                else:
                    cs_true += 1
                    cs_cnt += 1
                cs_true_list[int(commonSense)] += 1
                cs_cnt_list[int(commonSense)] += 1

                if uid[-1] == 'A':
                    pos_true += 1
                    pos_cnt += 1
                else:
                    neg_true += 1
                    neg_cnt += 1

            else:
                set_false += 1

                if c_len == 2:
                    c2_cnt += 1
                if c_len == 3:
                    c3_cnt += 1
                if c_len == 4:
                    c4_cnt += 1
                if c_len == 5:
                    c5_cnt += 1

                if commonSense == '0':
                    non_cs_cnt += 1
                else:
                    cs_cnt += 1
                cs_cnt_list[int(commonSense)] += 1

                if uid[-1] == 'A':
                    pos_cnt += 1
                    print(uid)
                else:
                    neg_cnt += 1
                    print(uid)

            set_cnt += 1

    print(f'set_true: {set_true}, set_false: {set_false}, set_cnt: {set_cnt}')
    print(f'avg: {set_true / set_cnt}\n')

    print(f'c2_acc: {c2_true / c2_cnt}')
    print(f'c3_acc: {c3_true / c3_cnt}')
    print(f'c4_acc: {c4_true / c4_cnt}')
    print(f'c5_acc: {c5_true / c5_cnt}\n')

    print(f'cs_acc: {cs_true / cs_cnt}')
    print(f'non_cs_acc: {non_cs_true / non_cs_cnt}\n')
    
if __name__ == '__main__':
    main()