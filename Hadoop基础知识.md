# Hadoop基础知识

> master/slave

hadoop的集群是基于master/slave模式。NameNode和JobTracker属于master，DataNode和TaskTracker属于slave。master只有一个，而slave有多个。

> NameNode

NameNode是中心服务器（单一节点），负责**管理文件系统的名称空间**和**客户端对文件的访问**

NameNode是主节点，存储文件的元数据，如文件名，文件目录结构，文件属性（生成时间，副本数，文件权限）以及每个文件的块列表，以及块所在的DataNode等等。

副本存放在哪些DataNode上由NameNode来控制，读取文件时NameNode尽量让用户先读取最近的副本，降低带块消耗和读取延时。

Namenode全权管理数据块的复制，它周期性地从集群中的每个Datanode接收心跳信号和块状态报告。接收到心跳信号意味着该Datanode节点工作正常。块状态报告包含了一个该Datanode上所有数据块的列表。

NameNode负责文件元数据的操作，DataNode负责文件内容读写请求。文件内容相关的数据不会经过NameNode，只会和DataNode联系，否则NameNode就会成为系统的瓶颈。

> DataNode

一个数据块在Datanode以**文件**存储在磁盘上，包括两个文件：**数据本身**和**元数据**(包括数据块的长度，块数据的校验和，以及时间戳)。

DataNode启动后向NameNode注册，通过后，周期性（1小时）的向NameNode上报所有的块信息。

心跳是每3秒一次，心跳返回结果带有Namenode给该Datanode的命令。如果超过10分钟没有收到某个Datanode的心跳，则认为该节点不可用。

DataNode存储数据的时候，都是以block形式存储，以块为单位，每个块有多个副本存储在不同的机器上，副本数可在文件生成时指定（默认3）。

block是DataNode存储数据的基本单位。默认Block的是128MB，这是每个Block的最大大小，而不是每个Block的大小都是128MB。

Datanode可以创建，删除，移动和重命名文件，当文件创建，写入和关闭之后不能修改文件的内容。

集群运行中可用安全加入和退出一些机器。