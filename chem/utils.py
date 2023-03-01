import statistics

def report(repeat, total_acc):
    assert len(total_acc) % repeat == 0
    num_task = int(len(total_acc) / repeat)
    task_std_list = []
    for t in range(num_task):
        task_acc = []
        for i in range(repeat):
            task_acc.append(total_acc[t + i * num_task])
        task_std = statistics.pstdev(task_acc)
        task_std_list.append(task_std)
    print('Average acc:{:.2f}±{:.2f}'.format(100 * sum(total_acc) / len(total_acc),
                                             100 * sum(task_std_list) / len(task_std_list)))