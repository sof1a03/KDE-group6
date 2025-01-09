import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-book-card',
  standalone: true,
  imports: [],
  templateUrl: './book-card.component.html',
  styleUrl: './book-card.component.css'
})

export class BookCardComponent {
  @Input() image_url = '';
  @Input() title= '';
  @Input() publisher= '';
  @Input() year= 0;
  @Input() ISBN= '';
  @Input() bookid= '';
  @Input() genres= [''];
}
