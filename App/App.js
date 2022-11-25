/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React, {useState} from 'react';
import ImagePicker from 'react-native-image-crop-picker';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  Dimensions,
  TouchableOpacity,
  View,
  Image,
} from 'react-native';

const App = () => {
  const [img, setImage] = useState();
  const handleImage = e => {
    console.log('Hello');
    ImagePicker.openPicker({
      mediaType: 'photo',
      width: 4000,
      height: 450,
      cropping: true,
    }).then(image => {
      setImage(image.path);
      let body = new FormData();
      body.append('image', {
        uri: image.path,
        name: 'image.jpeg',
        filename: 'imageName.jpeg',
        type: image.mime,
      });
      body.append('Content-Type', 'image/png');
      fetch('https://886a-42-110-136-7.ngrok.io', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
          otherHeader: 'foo',
        },
        body: body,
      })
        .then(res => res.json())
        .then(res => {
          console.log('response' + JSON.stringify(res));
        })
        .catch(e => console.log(e))
        .done();
    });
  };
  return (
    <SafeAreaView>
      <StatusBar
        barStyle={'dark-content'}
        // style={styles.backgroundColor}
      />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        style={styles.backgroundColor}>
        <View style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}>Machine Learning</Text>
          <Text style={styles.sectionDescription}>
            Personality Detection Based On Handwriting Analysis
          </Text>
          <TouchableOpacity style={styles.btn} onPress={handleImage}>
            <Text style={styles.btnText}>Take A Photo</Text>
          </TouchableOpacity>
          <Image
            style={{width: 300, height: 300, resizeMode: 'contain'}}
            source={{uri: img}}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  backgroundColor: {
    backgroundColor: '#0f001f',
    height: '100%',
  },
  sectionContainer: {
    margin: 10,
    paddingHorizontal: 24,
    backgroundColor: '#ebdcfc',
    height: Dimensions.get('window').height - 100,
    borderRadius: 5,

    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sectionTitle: {
    fontSize: 25,
    fontWeight: '300',
    color: '#0f001f',
  },
  sectionDescription: {
    fontSize: 12,
    fontWeight: '400',
  },
  highlight: {
    fontWeight: '700',
  },
  btn: {
    backgroundColor: '#0f001f',
    width: Dimensions.get('window').height / 3,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 35,
    marginTop: 20,
    marginBottom: 20,
  },
  btnText: {
    fontSize: 12,
    color: 'white',
  },
});

export default App;
