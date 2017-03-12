#! /pkg/bin/perl

while (<>) {
 chop;
 $c = $_;
 $filename = "/ldc/ldcsl/ctimit/$_.phn";
 $a =  `cat $filename`;
 split(/\n/,$a);
 $b = "";
 foreach (@_) {
  @e = split;
  $b .= "@e[2] @e[0] @e[1] ";
 }
 print "$c\t$b\n";
}
