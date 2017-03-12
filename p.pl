#! /pkg/bin/perl 


while (<>) {
 chop;
 if (/^([0-9]*)[^ ]CD $/) {
  print $1;
 } else {
  print "\t$_\n";
 }
}
