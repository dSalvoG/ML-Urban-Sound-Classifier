#!/usr/bin/env python

import os
import subprocess
import re
import shutil


def get_statistics():
    statistics = {}
    matcher = re.compile('\d+')
    # Top command on mac displays and updates sorted information about processes.

    top_command = subprocess.run(['top', '-l 1', '-n 0'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

    # Get Physical and Logical CPU Count
    physical_and_logical_cpu_count = os.cpu_count()
    statistics['physical_and_logical_cpu_count'] = physical_and_logical_cpu_count
    """
    # Load average 
    # This is the average system load calculated over a given period of time of 1, 5 and 15 minutes.
    # In our case, we will show the load average over a period of 15 minutes.
    
    # The numbers returned by os.getloadavg() only make sense if
    # related to the number of CPU cores installed on the system.
    
    # Here we are converting the load average into percentage. The higher the percentage the higher the load
    """

    cpu_load = [x / os.cpu_count() * 100 for x in os.getloadavg()][-1]
    statistics['cpu_load'] = round(cpu_load)

    # Memory usage
    total_ram = subprocess.run(['sysctl', 'hw.memsize'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
    vmLines = vm.split('\n')

    wired_memory = (int(matcher.search(vmLines[6]).group()) * 4096) / 1024 ** 3
    free_memory = (int(matcher.search(vmLines[1]).group()) * 4096) / 1024 ** 3
    active_memory = (int(matcher.search(vmLines[2]).group()) * 4096) / 1024 ** 3
    inactive_memory = (int(matcher.search(vmLines[3]).group()) * 4096) / 1024 ** 3

    # Used memory = wired_memory + inactive + active

    statistics['ram'] = dict({
        'total_ram': int(matcher.search(total_ram).group()) / 1024 ** 3,
        'used_ram': round(wired_memory + active_memory + inactive_memory, 2),
    })

    # Disk usage
    # Get total disk size, used disk space, and free disk

    total, used, free = shutil.disk_usage("/")

    # Number of Read and write operations
    # from the top command, the read written result will be as follows
    # 'Disks: XXXXXX/xxG read, XXXX/xxG written.'
    # we thus need to extract the read and written from this.
    read_written = top_command[9].split(':')[1].split(',')
    read = read_written[0].split(' ')[1]
    written = read_written[1].split(' ')[1]

    statistics['disk'] = dict(
        {
            'total_disk_space': round(total / 1024 ** 3, 1),
            'used_disk_space': round(used / 1024 ** 3, 1),
            'free_disk_space': round(free / 1024 ** 3, 1),
            'read_write': {
                'read': read,
                'written': written
            }
        }
    )

    # Network latency
    """
    Here we will ping google at an interval of five seconds for five times and record the
    min response time, average response time, and the max response time. 
    """
    ping_result = subprocess.run(['ping', '-i 5', '-c 5', 'google.com'], stdout=subprocess.PIPE).stdout.decode(
        'utf-8').split('\n')

    min, avg, max = ping_result[-2].split('=')[-1].split('/')[:3]
    statistics['network_latency'] = dict(
        {
            'min': min.strip(),
            'avg': avg.strip(),
            'max': max.strip(),
        }
    )
    return statistics


statistics = get_statistics()

# Here we can send the statistics to a server or even locally store the statistics in a database.
