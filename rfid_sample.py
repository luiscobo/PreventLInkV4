from sllurp import llrp
from twisted.internet import reactor
import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)


def cb(tagReport):
    tags = tagReport.msgdict['RO_ACCESS_REPORT']['TagReportData']
    for tag in tags:
        if 'EPC-96' in tag:
            tag_ig = tag['EPC-96'].decode()
            logging.info("Tag encontrado: %s", tag_ig)


if __name__ == '__main__':
    factory_args = dict(
        duration=3,
        start_inventory=True,
        disconnect_when_done=True,
        reconnect=True,
        tag_content_selector={
            'EnableROSpecID': False,
            'EnableSpecIndex': False,
            'EnableInventoryParameterSpecID': False,
            'EnableAntennaID': False,
            'EnableChannelIndex': True,
            'EnablePeakRSSI': False,
            'EnableFirstSeenTimestamp': False,
            'EnableLastSeenTimestamp': True,
            'EnableTagSeenCount': True,
            'EnableAccessSpecID': False
        },
        impinj_tag_content_selector=None,
    )
    fac = llrp.LLRPClientFactory(**factory_args)
    fac.addTagReportCallback(cb)
    logging.info("Connecting...")
    reactor.connectTCP('192.168.0.51', llrp.LLRP_PORT, fac, timeout=3)
    logging.info("Running...")
    reactor.run()
