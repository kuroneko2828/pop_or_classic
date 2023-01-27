import get_score_data
import split_data
import train
import estimate


def main():
    get_score_data.get_score_data()
    print('finish getting data!')
    split_data.split_data()
    print('finish splitting data!')
    train.train()
    print('finish training!')
    estimate.estimate()
    print('finish estimating!')
    print('finish all process')
    return


if __name__ == '__main__':
    main()