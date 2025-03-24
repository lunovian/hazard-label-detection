from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler

keyword = "ghs hazard labels 2.3 toxic gases"
data_dir = "data\\2.3_toxic_gases"

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={"root_dir": data_dir},
)
filters = dict(
    size="large",
    color="orange",
    license="commercial,modify",
    date=((2017, 1, 1), (2017, 11, 30)),
)
google_crawler.crawl(
    keyword=keyword,
    filters=filters,
    offset=0,
    max_num=1000,
    min_size=(200, 200),
    max_size=None,
    file_idx_offset=0,
)

bing_crawler = BingImageCrawler(downloader_threads=4, storage={"root_dir": data_dir})
bing_crawler.crawl(keyword=keyword, filters=None, offset=0, max_num=1000)

baidu_crawler = BaiduImageCrawler(storage={"root_dir": data_dir})
baidu_crawler.crawl(
    keyword=keyword, offset=0, max_num=1000, min_size=(200, 200), max_size=None
)
