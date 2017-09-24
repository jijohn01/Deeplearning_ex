import tensorflow as tf
#build graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
#실행 세션 생성
sess = tf.Session()

print("sess.run(node1, node2):",sess.run([node1,node2]))
print("sess.run(node3:",sess.run(node3))

#placeholder : 노드를 placeholder 로 생성하면 feed_dict로 이용해서 값을 넘겨줌
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict={a:[1, 3], b:[2, 4]}))
