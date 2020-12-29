#!/usr/bin/env python3

import csv
import logging
import os
import argparse

from check_ip import ip, hostname, ja3_sslbl_check, dns_tranco_check
from dgaintel import get_prob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Define logger for cross appliction logging consistency
logger = logging.getLogger(__name__)

ip_domain_dict = {}

csv_header = ['dom_in_tranco_1m', 'dom_dga_prob', 'otx_status', 'otx_age', 'urlhaus_status',
            'urlhaus_age', 'ja3_urlhaus_status', 'ja3_urlhaus_age', 'tls_record_type', 'client_tls_ver',
            'message_len', 'handshake_type', 'handshake_version', 'handshake_len',
            'cs_len', 'ext_len', 'src_port', 'dst_port', 'cs_0000', 'cs_0001', 'cs_0002', 
            'cs_0003', 'cs_0004', 'cs_0005', 'cs_0006', 'cs_0007', 'cs_0008','cs_0009',
            'cs_000a', 'cs_000b', 'cs_000c', 'cs_000d', 'cs_000e', 'cs_000f', 'cs_0010',
            'cs_0011', 'cs_0012','cs_0013', 'cs_0014', 'cs_0015', 'cs_0016', 'cs_0017', 'cs_0018',
            'cs_0019', 'cs_001a', 'cs_001b', 'cs_001e', 'cs_001f', 'cs_0020', 'cs_0021', 'cs_0022',
            'cs_0023', 'cs_0024', 'cs_0025', 'cs_0026', 'cs_0027', 'cs_0028', 'cs_0029',
            'cs_002a', 'cs_002b', 'cs_002c', 'cs_002d', 'cs_002e', 'cs_002f', 'cs_0030',
            'cs_0031', 'cs_0032', 'cs_0033', 'cs_0034', 'cs_0035', 'cs_0036', 'cs_0037',
            'cs_0038', 'cs_0039', 'cs_003a', 'cs_003b', 'cs_003c', 'cs_003d', 'cs_003e',
            'cs_003f', 'cs_0040', 'cs_0041', 'cs_0042', 'cs_0043', 'cs_0044', 'cs_0045',
            'cs_0046', 'cs_0067', 'cs_0068', 'cs_0069', 'cs_006a', 'cs_006b', 'cs_006c',
            'cs_006d', 'cs_0084', 'cs_0085', 'cs_0086', 'cs_0087', 'cs_0088', 'cs_0089',
            'cs_008a', 'cs_008b', 'cs_008c', 'cs_008d', 'cs_008e', 'cs_008f', 'cs_0090',
            'cs_0091', 'cs_0092', 'cs_0093', 'cs_0094', 'cs_0095', 'cs_0096', 'cs_0097',
            'cs_0098', 'cs_0099', 'cs_009a', 'cs_009b', 'cs_009c', 'cs_009d', 'cs_009e',
            'cs_009f', 'cs_00a0', 'cs_00a1', 'cs_00a2', 'cs_00a3', 'cs_00a4', 'cs_00a5',
            'cs_00a6', 'cs_00a7', 'cs_00a8', 'cs_00a9', 'cs_00aa', 'cs_00ab', 'cs_00ac',
            'cs_00ad', 'cs_00ae', 'cs_00af', 'cs_00b0', 'cs_00b1', 'cs_00b2', 'cs_00b3',
            'cs_00b4', 'cs_00b5', 'cs_00b6', 'cs_00b7', 'cs_00b8', 'cs_00b9', 'cs_00ba',
            'cs_00bb', 'cs_00bc', 'cs_00bd', 'cs_00be', 'cs_00bf', 'cs_00c0', 'cs_00c1',
            'cs_00c2', 'cs_00c3', 'cs_00c4', 'cs_00c5', 'cs_00c6', 'cs_00c7', 'cs_00ff',
            'cs_1301', 'cs_1302', 'cs_1303', 'cs_1304', 'cs_1305', 'cs_5600', 'cs_c001',
            'cs_c002', 'cs_c003', 'cs_c004', 'cs_c005', 'cs_c006', 'cs_c007', 'cs_c008',
            'cs_c009', 'cs_c00a', 'cs_c00b', 'cs_c00c', 'cs_c00d', 'cs_c00e', 'cs_c00f',
            'cs_c010', 'cs_c011', 'cs_c012', 'cs_c013', 'cs_c014', 'cs_c015', 'cs_c016',
            'cs_c017', 'cs_c018', 'cs_c019', 'cs_c01a', 'cs_c01b', 'cs_c01c', 'cs_c01d',
            'cs_c01e', 'cs_c01f', 'cs_c020', 'cs_c021', 'cs_c022', 'cs_c023', 'cs_c024',
            'cs_c025', 'cs_c026', 'cs_c027', 'cs_c028', 'cs_c029', 'cs_c02a', 'cs_c02b',
            'cs_c02c', 'cs_c02d', 'cs_c02e', 'cs_c02f', 'cs_c030', 'cs_c031', 'cs_c032',
            'cs_c033', 'cs_c034', 'cs_c035', 'cs_c036', 'cs_c037', 'cs_c038', 'cs_c039',
            'cs_c03a', 'cs_c03b', 'cs_c03c', 'cs_c03d', 'cs_c03e', 'cs_c03f', 'cs_c040',
            'cs_c041', 'cs_c042', 'cs_c043', 'cs_c044', 'cs_c045', 'cs_c046', 'cs_c047',
            'cs_c048', 'cs_c049', 'cs_c04a', 'cs_c04b', 'cs_c04c', 'cs_c04d', 'cs_c04e',
            'cs_c04f', 'cs_c050', 'cs_c051', 'cs_c052', 'cs_c053', 'cs_c054', 'cs_c055',
            'cs_c056', 'cs_c057', 'cs_c058', 'cs_c059', 'cs_c05a', 'cs_c05b', 'cs_c05c',
            'cs_c05d', 'cs_c05e', 'cs_c05f', 'cs_c060', 'cs_c061', 'cs_c062', 'cs_c063',
            'cs_c064', 'cs_c065', 'cs_c066', 'cs_c067', 'cs_c068', 'cs_c069', 'cs_c06a',
            'cs_c06b', 'cs_c06c', 'cs_c06d', 'cs_c06e', 'cs_c06f', 'cs_c070', 'cs_c071',
            'cs_c072', 'cs_c073', 'cs_c074', 'cs_c075', 'cs_c076', 'cs_c077', 'cs_c078',
            'cs_c079', 'cs_c07a', 'cs_c07b', 'cs_c07c', 'cs_c07d', 'cs_c07e', 'cs_c07f',
            'cs_c080', 'cs_c081', 'cs_c082', 'cs_c083', 'cs_c084', 'cs_c085', 'cs_c086',
            'cs_c087', 'cs_c088', 'cs_c089', 'cs_c08a', 'cs_c08b', 'cs_c08c', 'cs_c08d',
            'cs_c08e', 'cs_c08f', 'cs_c090', 'cs_c091', 'cs_c092', 'cs_c093', 'cs_c094',
            'cs_c095', 'cs_c096', 'cs_c097', 'cs_c098', 'cs_c099', 'cs_c09a', 'cs_c09b',
            'cs_c09c', 'cs_c09d', 'cs_c09e', 'cs_c09f', 'cs_c0a0', 'cs_c0a1', 'cs_c0a2',
            'cs_c0a3', 'cs_c0a4', 'cs_c0a5', 'cs_c0a6', 'cs_c0a7', 'cs_c0a8', 'cs_c0a9',
            'cs_c0aa', 'cs_c0ab', 'cs_c0ac', 'cs_c0ad', 'cs_c0ae', 'cs_c0af', 'cs_c0b0',
            'cs_c0b1', 'cs_c0b2', 'cs_c0b3', 'cs_c0b4', 'cs_c0b5', 'cs_c100', 'cs_c101',
            'cs_c102', 'cs_c103', 'cs_c104', 'cs_c105', 'cs_c106', 'cs_cca8', 'cs_cca9',
            'cs_ccaa', 'cs_ccab', 'cs_ccac', 'cs_ccad', 'cs_ccae', 'cs_d001', 'cs_d002',
            'cs_d003', 'cs_d005', 'cs_unknown', 'sig_0201', 'sig_0203', 'sig_0401', 'sig_0403', 'sig_0420',
            'sig_0501', 'sig_0503', 'sig_0520', 'sig_0601', 'sig_0603', 'sig_0620', 'sig_0704',
            'sig_0705', 'sig_0706', 'sig_0707', 'sig_0708', 'sig_0709', 'sig_070A', 'sig_070B',
            'sig_070C', 'sig_070D', 'sig_070E', 'sig_070F', 'sig_0804', 'sig_0805', 'sig_0806',
            'sig_0807', 'sig_0808', 'sig_0809', 'sig_080a', 'sig_080b', 'sig_081a', 'sig_081b',
            'sig_081c', 'sig_grease', 'sig_empty', 'grp_01', 'grp_02', 'grp_03', 'grp_04', 'grp_05',
            'grp_06', 'grp_07', 'grp_08', 'grp_09', 'grp_10', 'grp_11', 'grp_12', 'grp_13', 'grp_14',
            'grp_15', 'grp_16', 'grp_17', 'grp_18', 'grp_19', 'grp_20', 'grp_21', 'grp_22',
            'grp_23', 'grp_24', 'grp_25', 'grp_26', 'grp_27', 'grp_28', 'grp_29', 'grp_30',
            'grp_31', 'grp_32', 'grp_33', 'grp_34', 'grp_35', 'grp_36', 'grp_37', 'grp_38',
            'grp_39', 'grp_40', 'grp_41', 'grp_256', 'grp_257', 'grp_258', 'grp_259', 'grp_260',
            'grp_65281', 'grp_65282', 'grp_grease', 'pts_00', 'pts_01', 'pts_02', 'svr_ext_00', 'svr_ext_01',
            'svr_ext_02','svr_ext_03', 'svr_ext_04', 'svr_ext_05', 'svr_ext_06', 'svr_ext_07', 'svr_ext_08',
            'svr_ext_09', 'svr_ext_10', 'svr_ext_11', 'svr_ext_12', 'svr_ext_13', 'svr_ext_14',
            'svr_ext_15', 'svr_ext_16', 'svr_ext_17', 'svr_ext_18', 'svr_ext_19', 'svr_ext_20',
            'svr_ext_21', 'svr_ext_22', 'svr_ext_23', 'svr_ext_24', 'svr_ext_25', 'svr_ext_26',
            'svr_ext_27', 'svr_ext_28', 'svr_ext_29', 'svr_ext_30', 'svr_ext_31', 'svr_ext_32',
            'svr_ext_33', 'svr_ext_34', 'svr_ext_35', 'svr_ext_36', 'svr_ext_37', 'svr_ext_38',
            'svr_ext_39', 'svr_ext_40', 'svr_ext_41', 'svr_ext_42', 'svr_ext_43', 'svr_ext_44',
            'svr_ext_45', 'svr_ext_46', 'svr_ext_47', 'svr_ext_48', 'svr_ext_49', 'svr_ext_50',
            'svr_ext_51', 'svr_ext_52', 'svr_ext_53', 'svr_ext_55', 'svr_ext_56', 'svr_ext_65281',
            'svr_ext_unassigned', 'svr_ocsp_staple', 'svr_tls_ver', 'svr_supported_ver', 'malware_label']

# Create custom logging class for exceptions
class OneLineExceptionFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return repr(result)
 
    def format(self, record):
        result = super().format(record)
        if record.exc_text:
            result = result.replace("\n", "")
        return result

def write_csv_file(filename, data, header=False):
    '''
    Write data to csv
    '''
    if header:
        try:
            with open(filename, "w", newline='') as outfile:
                write_csv = csv.DictWriter(outfile, fieldnames=csv_header)
                write_csv.writeheader()
        except Exception as e:
            logging.exception("There was a problem in the CSV file write process... {}".format(e))
            exit(1)
    else:
        try:
            with open(filename, "a", newline='') as outfile:
                write_csv = csv.DictWriter(outfile, fieldnames=csv_header)
                write_csv.writerow(data)
        except Exception as e:
            logging.exception("There was a problem in the CSV file write process... {}".format(e))
            exit(1)

def correlate_data(csv_filename, tls_server_list, malware_label, API_KEY, out_dir, tls_client_entry):
    '''
    Reads in data from netcap TLS files and returns dictionary to insert into CSV file
    '''
    test_train_data = {}
    global ip_domain_dict
    ip_domain_value = ""
    tls_osint_list = []
    tranco_cache_dir = os.path.join(out_dir, '.tranco')

    # Pre-generate test_train_data_dict with 0 values
    for val in csv_header:
        test_train_data[val] = 0

    test_train_data['malware_label'] = malware_label

    for tls_server_data in tls_server_list:
        if tls_server_data['SrcIP'] == tls_client_entry['DstIP'] and tls_server_data['DstIP'] == tls_client_entry['SrcIP'] and tls_server_data['DstPort'] == tls_client_entry['SrcPort'] and tls_server_data['SrcPort'] == tls_client_entry['DstPort']:
            tls_server_entry = tls_server_data
            ip_domain_value = "{}:{}".format(tls_client_entry['DstIP'], tls_client_entry['SNI'])
            break
    
    # Check to see if key is in the ip_domain_dict (meaning it is an existing entry)
    # If it is, then skip check_ip, otherwise, we'll run it and add the values to the dict
    if ip_domain_value in ip_domain_dict.keys():
        tls_osint_list.append(ip_domain_dict[ip_domain_value][0])
        tls_osint_list.append(ip_domain_dict[ip_domain_value][1])
        tls_osint_list.append(ip_domain_dict[ip_domain_value][2])
        tls_osint_list.append(ip_domain_dict[ip_domain_value][3])
        tls_osint_list.append(ip_domain_dict[ip_domain_value][4])
    else:
        dst_ip = tls_client_entry['DstIP']
        sni = tls_client_entry['SNI']
        tls_osint_list.append(ip(API_KEY, dst_ip))
        tls_osint_list.append(hostname(sni, dst_ip))
        tls_osint_list.append(ja3_sslbl_check(tls_client_entry['Ja3']))

        # Tranco and dgaintel fail when domain name is empty
        if not sni == '':
            tls_osint_list.append(dns_tranco_check(tranco_cache_dir, sni, 15))
            tls_osint_list.append(get_prob(sni))
        else:
            tls_osint_list.append(0)
            tls_osint_list.append(0)

        ip_domain_dict[ip_domain_value] = tls_osint_list

    # OSINT OTX and urlhaus analysis
    test_train_data['otx_status'] = tls_osint_list[0]['url_status']
    test_train_data['otx_age'] = tls_osint_list[0]['report_age']
    test_train_data['urlhaus_status'] = tls_osint_list[1]['url_status']
    test_train_data['urlhaus_age'] = tls_osint_list[1]['report_age']
    test_train_data['ja3_urlhaus_status'] = tls_osint_list[2]['ja3_check']
    test_train_data['ja3_urlhaus_age'] = tls_osint_list[2]['ja3_record_age']
    test_train_data['dom_in_tranco_1m'] = tls_osint_list[3]
    test_train_data['dom_dga_prob'] = tls_osint_list[4]

    # Set TLS Client static data fields in test_train_data dict
    test_train_data['tls_record_type'] = tls_client_entry['Type']
    test_train_data['client_tls_ver'] = tls_client_entry['Version']
    test_train_data['message_len'] = tls_client_entry['MessageLen']
    test_train_data['handshake_type'] = tls_client_entry['HandshakeType']
    test_train_data['handshake_version'] = tls_client_entry['HandshakeVersion']
    test_train_data['handshake_len'] = tls_client_entry['HandshakeLen']
    test_train_data['cs_len'] = tls_client_entry['CipherSuiteLen']
    test_train_data['ext_len'] = tls_client_entry['ExtensionLen']
    test_train_data['src_port'] = tls_client_entry['SrcPort']
    test_train_data['dst_port'] = tls_client_entry['DstPort']

    # Set TLS Server static data fields in test_train_data dict
    test_train_data['svr_tls_ver'] = tls_server_entry['Version']
    test_train_data['svr_supported_ver'] = tls_server_entry['SupportedVersion']
    if tls_server_entry['OCSPStapling'] == 'false':
        test_train_data['svr_ocsp_staple'] = 0
    else:
        test_train_data['svr_ocsp_staple'] = 1

    svr_selected_group = "{:02}".format(int(tls_server_entry['SelectedGroup']))
    server_cs_used = "{:04x}".format(int(tls_server_entry['CipherSuite']))

    try:
        # Cipher Suites
        tls_client_entry['CipherSuites'] = tls_client_entry['CipherSuites'][1:-1].split('-')
        for cs_val in tls_client_entry['CipherSuites']:
            entry_hex = "{:04x}".format(int(cs_val))

            cs_entry = "cs_{}".format(entry_hex)

            if cs_entry in test_train_data:
                test_train_data[cs_entry] += 0.5
            else:
                test_train_data['cs_unknown'] += 0.5
            
            if entry_hex == server_cs_used:
                test_train_data[cs_entry] += 0.5

        # Signature Algorithms
        tls_client_entry['SignatureAlgs'] = tls_client_entry['SignatureAlgs'][1:-1].split('-')

        if not tls_client_entry['SignatureAlgs'][0]:
            tls_client_entry['SignatureAlgs'] = ['0']

        sig_reserved_count = 1
        for sig_data in tls_client_entry['SignatureAlgs']:
            sig_entry = "sig_{:04x}".format(int(sig_data))

            if sig_entry in test_train_data:
                test_train_data[sig_entry] = 1
            elif sig_entry == 'sig_0000':
                test_train_data['sig_empty'] = 1
            else:
                test_train_data['sig_grease'] = sig_reserved_count
                sig_reserved_count += 1

        # Supported Groups
        tls_client_entry['SupportedGroups'] = tls_client_entry['SupportedGroups'][1:-1].split('-')
        for grp_data in tls_client_entry['SupportedGroups']:
            grp_val = "{:02}".format(int(grp_data))

            grp_entry = "grp_{}".format(grp_val)

            if grp_val in test_train_data:
                test_train_data[grp_entry] = 0.5
            else:
                grp_entry = 'grp_grease'
                test_train_data[grp_entry] += 0.5

            if grp_val == svr_selected_group:
                test_train_data[grp_entry] += 0.5

        # Supported Points
        if tls_client_entry['SupportedPoints'][0] == '(':
            tls_client_entry['SupportedPoints'] = tls_client_entry['SupportedPoints'][1:-1].split('-')
        else:
            tls_client_entry['SupportedPoints'] = [tls_client_entry['SupportedPoints']]

        for pts_data in tls_client_entry['SupportedPoints']:
            pts_entry = "pts_{:02}".format(int(pts_data))
            test_train_data[pts_entry] = 1

        # Server Extensions
        if tls_server_entry['Extensions'][0] == '(':
            tls_server_entry['Extensions'] = tls_server_entry['Extensions'][1:-1].split('-')
        else:
            tls_server_entry['Extensions'] = [tls_server_entry['Extensions'][1:]]

        if tls_server_entry['Extensions'][0] == '':
            tls_server_entry['Extensions'] = ['0']

        for svr_ext_data in tls_server_entry['Extensions']:
            svr_ext_entry = "svr_ext_{:02}".format(int(svr_ext_data))

            if svr_ext_entry in test_train_data:
                test_train_data[svr_ext_entry] = 1
            else:
                test_train_data['svr_ext_unassigned'] += 0.5

    except Exception as e:
        print("The problem is with a loop...  -  {}".format(e))

    write_csv_file(csv_filename, test_train_data)    

def main():
    '''
    Gather and format data from the TLSClientHello and TLSServerHello CSV files generated from NetCap
    '''
    # Get a label value for known data
    parser = argparse.ArgumentParser(description='Get label input for data analysis')
    parser.add_argument('-l', '--label', action='store', dest='label', default=0, help='Is this data known malicious or not?', required=False)
    parser.add_argument('-o', '--outfile', action='store', dest='out_file', default='test_train_data.csv',
                        help='Name of the output file', required=False)
    parser.add_argument('-c', '--client-file', action='store', dest='client_file', default='TLSClientHello.csv',
                        help='Name of the TLS Client Hello (TLSClientHello.csv) file created with NetCap', required=False)
    parser.add_argument('-s', '--server-file', action='store', dest='server_file', default='TLSServerHello.csv',
                        help='Name of the TLS Server Hello (TLSServerHello.csv) file created with NetCap', required=False)
    parser.add_argument('-a', '--api-key', action='store', dest='api', default='<Enter your Alienvault OTX API Key here',
                        help='API Key value required for Alienvault OTX', required=False)

    options = parser.parse_args()

    base_log_dir = os.getcwd()
    out_dir = r'C:\Users\bryan\Desktop'
    csv_filename = os.path.join(out_dir, options.out_file)
    tls_client_file = os.path.join(base_log_dir, options.client_file)
    tls_server_file = os.path.join(base_log_dir, options.server_file)
    tls_server_list = []
    malicious_label = options.label
    #API_KEY = os.environ.get('API_KEY')
    API_KEY = options.api

    if not os.path.exists(csv_filename):
        write_csv_file(csv_filename, csv_header, True)

    with open(tls_server_file, 'r', newline='') as tls_server_data:
        tls_server_csv = csv.DictReader(tls_server_data)
        for line in tls_server_csv:
            tls_server_list.append(line)

    with ProcessPoolExecutor() as executor:
        fn = partial(correlate_data, csv_filename, tls_server_list, malicious_label, API_KEY, out_dir)
        with open(tls_client_file, 'r', newline='') as tls_client_data:
            tls_client_csv = csv.DictReader(tls_client_data)
            executor.map(fn, tls_client_csv, timeout=86400)

if __name__ == '__main__':
    try:
        exit(main())
    except Exception:
        logging.exception("Exception in main()")
        exit(1)