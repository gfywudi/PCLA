import argparse


def get_train_config():
        parse = argparse.ArgumentParser(description='student information model')
        # training parameters
        parse.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parse.add_argument('-model_name', type=str, default='transforemer model', help='model name')
        parse.add_argument('-data-path', type=str, default='./stu_encode2.csv', help='dataset path')
        parse.add_argument('-fold_lr', type=float, default=0.001, help='fold train learning rate')
        parse.add_argument('-final_lr', type=float, default=0.00005, help='final train learning rate')


        parse.add_argument('-reg', type=float, default=1e-5, help='weight lambda of regularization')
        parse.add_argument('-batch-size', type=int, default=32, help='number of samples in a batch')
        parse.add_argument('-random_seed', type=int, default=0, help='learning rate')
        # parse.add_argument('-batch-size', type=int, default=16, help='number of samples in a batch')
        parse.add_argument('-epoch', type=int, default=70, help='number of iteration')
        parse.add_argument('--k-fold', type=int, default=2, help='k in cross validation')
        parse.add_argument('-data_dim', type=int, default=25, help='data_dim,(x, 5000/x)')

        parse.add_argument('-cuda', type=bool, default=True, help='if use cuda')
        parse.add_argument('-device', type=str, default='0', help='device id')
        parse.add_argument('-n_gpu', type=int, default=7, help='n_gpu to use')
        parse.add_argument('--lr_decay_epochs', type=str, default='7,15', help='where to decay lr, can be a list')
        parse.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        # cosine annealing
        parse.add_argument('--cosine', action='store_true', help='using cosine annealing')


        parse.add_argument('-interval-log', type=int, default=100,
                        help='how many batches have gone through to record the training performance')
        parse.add_argument('-interval-valid', type=int, default=1,
                        help='how many epoches have gone through to record the validation performance')
        parse.add_argument('-interval-test', type=int, default=1,
                        help='how many epoches have gone through to record the test performance')

        parse.add_argument('-signal-lead', type=str, default='1',
                help='how many lead have gone through to train model')

        parse.add_argument('-num_class', type=str, default='4',
                help='how many lead have gone through to train model')

        
        parse.add_argument('-n_way', type=int, default=4, help='number of iteration')
        parse.add_argument('-k_shot', type=int, default=4, help='number of iteration')
        parse.add_argument('-n_query', type=int, default=1, help='number of iteration')
        parse.add_argument('-test-episode', type=int, default=200, help='number of iteration')

        config = parse.parse_args()
        return config