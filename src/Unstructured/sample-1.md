Discoverability and Analytics



Progressive Enhancement

Discoverability and PWA’s

Progressive enhancement = content available to all users

JS sites are indexed by Google**

Test your site with Fetch as Google tool

**Follow these best practices



What is Google Analytics?

Data is generated from user behavior

Data is processed by Google Analytics back end

Reports are generated



Integration process overview	

Create account and “property”

Paste tracking snippet

(Optional) Add custom analytics



Accounts and properties

Account: 

My Company

Property #1: 

iOS app

Property #2: 

Android app

Property #3: 

Web app

Property ID: 12345-6

Property ID: 23456-7

Property ID: 34567-8



Adding analytics to your app

<script>

// [a bunch of uglified JS]

ga('create', 'UA-XXXXXXXX-Y', 'auto');

ga('send', 'pageview');

</script>



The Google Analytics dashboard



The Google Analytics dashboard (cont.)



Real-time analytics



Custom events

ga('send', {

  hitType: 'event',

  eventCategory: 'products',

  eventAction: 'view more',

  eventLabel: 'premium'

});

// Alternative syntax

ga('send', 'event', 'products', 'view more', 'premium');



Example: Push notifications

registration.pushManager.subscribe({

  userVisibleOnly: true

})

.then(function(pushSubscription) {

  ga('send', 'event', 'push', 'subscribe'); 

});



Analytics and service workers

The Analytics ga function requires Window

Service worker hits use the Measurement Protocol API:

POST /collect HTTP/1.1Host: www.google-analytics.comv=1&tid=UA-XXXXX-Y&cid=555&t=pageview&dp=%2Fhome



Measurement Protocol example

self.addEventListener('notificationclose', function(event) {

  event.waitUntil(

    fetch('https://www.google-analytics.com/collect', {

      method: 'post',

      body: 'v=1&cid=...&tid=UA-XXXXXXXX-Y&t=event&' +

            'ec=notification&ea=delay&el=serviceworker'

    });

  );

});

service-worker.js



What about when users are offline?

Install

$ npm install --save-dev sw-offline-google-analytics

Service worker

importScripts('path/to/offline-google-analytics-import.js');

goog.offlineGoogleAnalytics.initialize();



Analytics stored in IndexedDB



Lab Overview

Create a Google Analytics & Firebase account

Add analytics to an app

View analytics data

Add custom events to understand user behavior

Use analytics with service workers (& push notifications)

Use the Measurement Protocol API

Add offline analytics to an app



Resources

Resources

analytics.js 

Reporting API

Google Analytics Academy 

Account signup

Analytics for mobile applications

Chrome debugger extension

ImportScripts

Offline Google Analytics 

IndexedDB

Measuring Critical Performance Metrics with Google Analytics

Measurement Protocol

Google I/O offline example



UI reference slides



Finding your tracking snippet & ID

Select the Admin tab 

Under “account”, select your account from the drop down list. 

Then under “properties”, select your property from the down list. 

Now choose “tracking info”, followed by “tracking code”. 



Finding the Audience Overview interface

Select the Reporting tab

Select Audience in the side panel

Select Overview



Finding the Real-time Overview interface

Select the Reporting tab

Select Real-time in the side panel

Select Overview



Finding the Real-time Events interface

Select the Reporting tab

Select Real-time in the side panel

Select Events



Finding the Events Overview interface

Select the Reporting tab

Select Behavior in the side panel

Select Events 

Select Overview 