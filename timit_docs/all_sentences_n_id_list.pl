#! /pkg/bin/perl

while (<>) {
 chop;
 $filename = "$_.phn";
 `cat $filename >> tmp0`;
}
