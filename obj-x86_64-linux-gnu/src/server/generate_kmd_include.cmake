  execute_process (COMMAND /usr/bin/kurento-module-creator -r /home/parallels/src/kms-opencv-plugin-sample/src/server/interface -dr /usr/share/kurento/modules -o /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/)

  file (READ /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/opencvpluginsample.kmd.json KMD_DATA)

  string (REGEX REPLACE "\n *" "" KMD_DATA ${KMD_DATA})
  string (REPLACE "\"" "\\\"" KMD_DATA ${KMD_DATA})
  string (REPLACE "\\n" "\\\\n" KMD_DATA ${KMD_DATA})
  set (KMD_DATA "\"${KMD_DATA}\"")

  file (WRITE /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/opencvpluginsample.kmd.json ${KMD_DATA})
