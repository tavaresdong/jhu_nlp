from tensorflow.examples.tutorials.mnist import input_data

def download_mnist():
    # 下载mnist数据集
    mnist = input_data.read_data_sets('/tmp/', one_hot=True)


if __name__ == "__main__":
    download_mnist()