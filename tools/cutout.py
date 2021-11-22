from lib.core import CutOuter


def main():
    cutouter = CutOuter(root='../data/problem')
    cutouter.cutout(eps=0)


if __name__ == '__main__':
    main()
